from typing import Annotated, List, Sequence, TypedDict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field
from typing import Dict, Any
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
import os
import requests
from firecrawl import FirecrawlApp, ScrapeOptions
from bs4 import BeautifulSoup
import time

load_dotenv()


def random_uuid():
    return str(uuid4())


class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


class URLSelector(BaseModel):
    selected_urls: list[str] = Field(
        description="A list of URLs that are relevant to the restaurant and query.")


class Review(BaseModel):
    content: str = Field(
        description="A review from the website.")
    rate: int = Field(
        description="The rating of the review.")
    date: str = Field(
        description="The date of the review.")
    author: str = Field(
        description="The author of the review.")
    title: str = Field(
        description="The title of the review.")
    image_urls: list[str] = Field(
        description="The image urls of the review.")


class ReviewExtractor(BaseModel):
    reviews: list[Review] = Field(
        description="A list of reviews from the website.")


class SearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    restaurant_name: str
    prompt: str
    summary: str
    reviews: List[Review]


class Review(BaseModel):
    review_url: str
    review_text: str


class ReivewList(BaseModel):
    reviews: List[Review]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash")  # 또는 "gemini-1.5-pro" 등
llm_url_selector = llm.with_structured_output(URLSelector)
llm_review_extractor = llm.with_structured_output(ReviewExtractor)
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=firecrawl_api_key)

url_selector_system_message = SystemMessage(
    content="""
    You are a URL selector that always responds with valid JSON.
    You select URLs from the search results relevant to the restaurant, location, and query.
    Return a JSON object with a property 'selected_urls' that contains an array
    of URLs most likely to help meet the objective. Add a /* to the end of the URL if you think it should search all of the pages in the site. Do not return any social media links. For example: {\"selected_urls\": [\"https://example.com\", \"https://example2.com\"]} "
    You should not return any other text than the JSON object such as ```, json, etc.
    """)
url_selector_user_message = HumanMessage(
    content="""
    Restaurant: {restaurant_name}
    Location: {restaurant_location}
    Query: {query}
    HTML Results: {html_results}
    """)


review_extractor_system_message = SystemMessage(
    content="""
    You are a Review Extractor that extracts the reviews from the website. You will be given the HTML of the website and you will need to extract the reviews. It has reviewText on data_automation field.

    Additionally, extract the image urls from the <picture> element that contains one or more nested <source> or <img> elements with srcset attributes. Check all <source> and <img> tags inside the <picture> element. From their srcset attributes, select the image url with the highest resolution by finding the largest width value (for example, w=1200 or w=1100). If there is no srcset, use the src attribute of the <img> element instead.
    DO NOT COLLECT THUMBNAIL IMAGES.
    Return a JSON object with a property 'reviews' that contains an array of reviews. Each review should have a property 'content' that contains the review, a property 'rate' that contains the rating, a property 'date' that contains the date of the review, a property 'author' that contains the author of the review, a property 'title' that contains the title of the review, and a property 'image_url' that contains the extracted image url.
    """
)
review_extractor_user_message = HumanMessage(
    content="""
    HTML Results: {html_results}
    """)


def accumulate_reviews_fn(existing, new):
    return (existing or []) + new


class SearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    restaurant_name: str
    restaurant_location: str
    reviews: Annotated[List[Review], accumulate_reviews_fn]
    search_results: List[Dict[str, Any]]
    review_urls: List[str]
    search_domain: str
    initial_review_url: str
    pagination_pattern: str
    query_review_url: str


# Not used
def crawl_urls(url):
    """Crawl the URLs and return the data."""
    response = app.crawl_url(
        url=url,
        max_depth=1,
        max_discovery_depth=1,
        allow_external_links=False,
        allow_backward_links=False,
        limit=1,
        scrape_options=ScrapeOptions(
            formats=['html'],
            onlyMainContent=True
        )
    )

    return response


def scrape_url(url: str) -> str:
    response = app.scrape_url(
        url=url,
        formats=['html'],
        only_main_content=True,
        wait_for=5000,
        timeout=20000
    )

    return response


def extract_reviews_div(html_content: str) -> str:
    # print(
    #     f"{Colors.GREEN}Extracting reviews div from {html_content[:100]}{Colors.RESET}")
    with open("doc.html", "w") as f:
        f.write(html_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    # id='Reviews' 우선 검색
    reviews_div = soup.find(lambda tag: tag.name ==
                            'div' and tag.get('id', '').lower() == 'reviews')

    if reviews_div:
        return str(reviews_div)
    else:
        print("Reviews div not found.")
        return ""


def scrape_url_with_retry(url: str) -> str:
    max_retries = 3
    for i in range(max_retries):
        try:
            response = scrape_url(url)
        except Exception as e:
            print(
                f"{Colors.RED}Error scraping URL: {e} {url} {i} {max_retries}{Colors.RESET}")
            time.sleep(1)
            continue

        extracted_html = extract_reviews_div(response.html)
        if extracted_html:
            print(
                f"{Colors.GREEN}Success in scrape_url_with_retry {url}.{Colors.RESET}")
            return extracted_html
        else:
            print(f"{Colors.RED}Fail in scrape_url_with_retry {url}.{Colors.RESET}")
            time.sleep(1)
    print(f"{Colors.RED}Failed to scrape URL after {max_retries} attempts. {url} {Colors.RESET}")
    return ""


def search_google(state: SearchState) -> SearchState:
    def request_google_search(query: str) -> List[str]:
        url = 'https://customsearch.googleapis.com/customsearch/v1'

        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CSE_ID")

        params = {
            'q': query,
            'cx': cx,
            'key': api_key
        }

        headers = {
            'Accept': 'application/json'
        }

        response = requests.get(url, params=params, headers=headers)
        return response.json()

    query = f"{state['restaurant_name']} {state['restaurant_location']} site:{state['search_domain']}"
    data = request_google_search(query)
    if data.get("items", []):
        initial_review_url = data.get("items", [])[0].get("link")
        return {"initial_review_url": initial_review_url}
    else:
        return {"initial_review_url": None}


def extract_review_urls(state: SearchState) -> SearchState:
    initial_review_url = state["initial_review_url"]

    if initial_review_url:
        print(f"{Colors.CYAN}Initial review URL: {initial_review_url}{Colors.RESET}")
        reviews_div = scrape_url_with_retry(initial_review_url)
    else:
        return {"review_urls": []}

    try:

        for _ in range(3):
            # Prepare the data for R1
            print(
                f"{Colors.CYAN}Extracting review urls for {reviews_div[:100]}{Colors.RESET}")
            query = f"Search {state['restaurant_name']} {state['restaurant_location']} pagination links such that {state['pagination_pattern']} from {state['search_domain']}"
            response = llm_url_selector.invoke([
                url_selector_system_message,
                url_selector_user_message.content.format(
                    restaurant_name=state["restaurant_name"],
                    restaurant_location=state["restaurant_location"],
                    query=query,
                    html_results=reviews_div
                )
            ])
            print(f"{Colors.GREEN}Response: {response}{Colors.RESET}")

            urls = response.selected_urls

            # Clean up URLs - remove wildcards and trailing slashes
            cleaned_urls = [url.replace('/*', '').rstrip('/') for url in urls]
            cleaned_urls = [url for url in cleaned_urls if url]
            if cleaned_urls:
                return {"review_urls": cleaned_urls}
            print(f"{Colors.YELLOW}No valid URLs found. Retrying...{Colors.RESET}")
            time.sleep(1)

        print(f"{Colors.YELLOW}No valid URLs found.{Colors.RESET}")
        return {"review_urls": []}

    except Exception as e:
        print(f"{Colors.RED}Error selecting URLs with R1: {e}{Colors.RESET}")
        return {"review_urls": []}


# Rule-based pagination URL generator
def generate_pagination_urls(initial_url: str, max_pages: int = 5) -> list:
    """
    Generate a list of paginated URLs from the initial review URL using the 'or' pattern.
    E.g., for Tripadvisor: ...-Reviews-...html, generate ...-Reviews-or15-...html, ...-Reviews-or30-...html, etc.
    """
    urls = [initial_url]
    import re
    # Find insertion point for pagination, e.g., between 'Reviews' and the next '-'
    # Example: https://www.tripadvisor.com/Restaurant_Review-g60750-d491264-Reviews-Filippi_s_Pizza_Grotto_Little_Italy-San_Diego_California.html
    # Pagination: ...-Reviews-or15-...
    match = re.search(r"(Reviews)(-)", initial_url)
    if not match:
        return urls
    base = initial_url[:match.end(1)]
    rest = initial_url[match.end(1):]
    for page in range(1, max_pages):
        offset = 15 * page
        paginated_url = f"{base}-or{offset}{rest}"
        urls.append(paginated_url)
    return urls


def continue_crawl(state: SearchState) -> SearchState:
    # Use rule-based pagination URL generator
    initial_url = state.get("initial_review_url")
    if not initial_url:
        return []
    paginated_urls = generate_pagination_urls(initial_url, max_pages=5)
    send_nodes = [Send("extract_reviews", {
                       "query_review_url": url}) for url in paginated_urls if url]
    return send_nodes


def extract_reviews(state: SearchState) -> SearchState:
    query_review_url = state["query_review_url"]

    reviews_div = scrape_url_with_retry(query_review_url)
    if not reviews_div:
        print(f"{Colors.RED}No reviews div found in {query_review_url}.{Colors.RESET}")
        return {"reviews": []}

    try:
        response = llm_review_extractor.invoke([
            review_extractor_system_message,
            review_extractor_user_message.content.format(
                html_results=reviews_div
            )
        ])
        return {"reviews": response.reviews}
    except Exception as e:
        print(f"{Colors.RED}Error extracting reviews with R2: {e}{Colors.RESET}")
        return {"reviews": []}


graph = StateGraph(SearchState)
graph.set_entry_point("search google")
graph.add_node("search google", search_google)
graph.add_node("extract_review_urls", extract_review_urls)
graph.add_node("extract_reviews", extract_reviews)

graph.add_edge("search google", "extract_review_urls")
graph.add_conditional_edges("extract_review_urls", continue_crawl)
graph.add_edge("extract_reviews", END)

search_graph = graph.compile()
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# Restaurant review: pagination pattern is /Restaurant_Review.*-or\d+-.*\.html$
# ret = search_graph.invoke({"messages": [],
#                           "restaurant_name": "Filippi's Pizza Grotto Little Italy",
#                            "restaurant_location": "San Diego, CA",
#                            "search_domain": "tripadvisor.com",
#                            "pagination_pattern": "^/Restaurant_Review.*-or\\d+-.*\\.html$"})

# Hotel review: pagination pattern is /Hotel_Review.*-or\d+-.*\.html$
ret = search_graph.invoke({"messages": [],
                          "restaurant_name": "Fairmont Grand Del Mar",
                           "restaurant_location": "San Diego, CA",
                           "search_domain": "tripadvisor.com",
                           "pagination_pattern": "^/Hotel_Review.*-or\\d+-.*\\.html$"})
