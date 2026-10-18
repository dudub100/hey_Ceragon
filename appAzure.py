import streamlit as st
import os
import base64
import time
import hashlib
from urllib.parse import urlparse
import fitz  # PyMuPDF
import math
import numpy as np
from scipy.optimize import root_scalar
import astropy.units as u
import itur

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    ScalarQuantizationCompression,
    ScalarQuantizationParameters,
)
from azure.search.documents.models import VectorizedQuery

# --- Page Configuration ---
st.set_page_config(page_title="VX/IP-Series Hybrid Telecom Assistant", layout="centered")

# --- Secrets: environment variables (Azure App Service) or st.secrets (Streamlit Cloud) ---
def get_secret(key, default=None):
    value = os.environ.get(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default

def load_users():
    """
    Azure App Service: env var APP_USERS = "User1=password,technician;User2=password,customer"
    Streamlit Cloud:   [users] section in secrets, e.g. User1 = "password,technician"
    """
    env_users = os.environ.get("APP_USERS")
    if env_users:
        users = {}
        for entry in env_users.split(";"):
            if "=" in entry:
                name, value = entry.split("=", 1)
                users[name.strip()] = value.strip()
        return users
    try:
        return dict(st.secrets["users"])
    except Exception:
        return {}

USER_DB = load_users()

# OData filters applied to every search, per role
ROLE_FILTERS = {
    "customer": "role eq 'customer'",
    "technician": "search.in(role, 'customer,technician', ',')",
}

# --- Setup Keys ---
_missing = [k for k in ("GOOGLE_API_KEY", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY") if not get_secret(k)]
if _missing:
    st.error(f"Missing configuration: {', '.join(_missing)}. Add them to the app secrets / environment variables.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY")

AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_secret("AZURE_SEARCH_KEY")
INDEX_NAME = get_secret("AZURE_SEARCH_INDEX", "manuals-index")
EMBEDDING_DIM = 768

# --- Authentication Logic ---
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

def login_ui():
    st.title("Secure Technical Portal")
    if not USER_DB:
        st.error("No users configured. Add a [users] section to the app secrets.")
        st.stop()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_db_lower = {k.lower(): v for k, v in USER_DB.items()}
        username_lower = username.strip().lower()

        if username_lower in user_db_lower:
            saved_password, saved_role = user_db_lower[username_lower].split(",")
            if password == saved_password.strip():
                st.session_state['user_role'] = saved_role.strip()
                st.rerun()
            else:
                st.error("Invalid password.")
        else:
            st.error("Username not found.")

# --- Microsoft SSO (Azure App Service Authentication / "Easy Auth") ---
def get_sso_user():
    """
    Returns the signed-in Microsoft account (e.g. name@ceragon.com) when running on
    Azure App Service with Authentication enabled. App Service removes these headers
    from incoming requests and sets them itself, so they can be trusted there.
    Outside Azure (e.g. Streamlit Cloud) they are ignored.
    """
    if not os.environ.get("WEBSITE_SITE_NAME"):  # set automatically by App Service
        return None
    try:
        return st.context.headers.get("X-Ms-Client-Principal-Name")
    except Exception:
        return None

def role_for_sso_user(email: str) -> str:
    """Every authenticated Ceragon user is a customer, unless listed in TECHNICIAN_USERS."""
    technicians = {e.strip().lower() for e in (get_secret("TECHNICIAN_USERS", "") or "").split(",") if e.strip()}
    return "technician" if email.lower() in technicians else "customer"

sso_user = get_sso_user()
if sso_user and not st.session_state['user_role']:
    st.session_state['user_role'] = role_for_sso_user(sso_user)
    st.session_state['user_name'] = sso_user

if not st.session_state['user_role']:
    login_ui()
    st.stop()

# --- Sidebar Controls & Model Selection ---
if sso_user:
    st.sidebar.success(f"Signed in as {sso_user} ({st.session_state['user_role'].capitalize()})")
else:
    st.sidebar.success(f"Logged in as: {st.session_state['user_role'].capitalize()}")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Settings")

MODEL_OPTIONS = {
    "GPT-OSS 120B (Groq - Deep Logic & Telecom, Free)": "groq:openai/gpt-oss-120b",
    "Gemini 2.5 Flash (Google)": "gemini:gemini-2.5-flash",
    "GPT-OSS 20B (Groq - Lightning Fast, Free)": "groq:openai/gpt-oss-20b",
    "Auto-Free Open Source (OpenRouter)": "openrouter:openrouter/free",
    "Llama 3 8B (OpenRouter - Free)": "openrouter:meta-llama/llama-3.1-8b-instruct:free"
}

# Azure (Microsoft Foundry) models: one dropdown entry per deployment name in AZURE_AI_DEPLOYMENTS
_azure_deployments = [d.strip() for d in (get_secret("AZURE_AI_DEPLOYMENTS", "") or "").split(",") if d.strip()]
if _azure_deployments and get_secret("AZURE_AI_ENDPOINT") and get_secret("AZURE_AI_KEY"):
    MODEL_OPTIONS = {
        **{f"{d} (Azure Foundry)": f"azure:{d}" for d in _azure_deployments},
        **MODEL_OPTIONS,
    }

selected_model_label = st.sidebar.selectbox("Select Reasoning / Chat Model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

if sso_user:
    st.sidebar.markdown("[Sign out](/.auth/logout)")
elif st.sidebar.button("Logout"):
    st.session_state['user_role'] = None
    st.rerun()

# --- Model Initialization Factories ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

@st.cache_resource
def get_embedding_engine():
    """768-dimension Gemini embeddings (must match EMBEDDING_DIM in the index)."""
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=EMBEDDING_DIM)

@st.cache_resource
def get_vision_llm():
    """Dedicated Multimodal Vision Engine for PDF Processing."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, safety_settings=SAFETY_SETTINGS)

def get_chat_llm(selected_value):
    """Instantiates the user-selected Chat Reasoning model across multiple providers."""
    provider, model_id = selected_value.split(":", 1)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model_id, temperature=0, safety_settings=SAFETY_SETTINGS)

    elif provider == "groq":
        groq_key = get_secret("GROQ_API_KEY")
        if not groq_key:
            st.error("⚠️ GROQ_API_KEY missing in secrets! Defaulting to Gemini.")
            return get_chat_llm("gemini:gemini-2.5-flash")
        return ChatGroq(groq_api_key=groq_key, model_name=model_id, temperature=0.01, max_tokens=4096)

    elif provider == "openrouter":
        or_key = get_secret("OPENROUTER_API_KEY")
        if not or_key:
            st.error("⚠️ OPENROUTER_API_KEY missing in secrets! Defaulting to Gemini.")
            return get_chat_llm("gemini:gemini-2.5-flash")
        return ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key,
                          model=model_id, temperature=0.01, max_tokens=4096)

    elif provider == "azure":
        # Uses the Azure OpenAI v1 (OpenAI-compatible) endpoint of the Foundry resource.
        # Any URL from the resource works: only the host is kept.
        parsed = urlparse(get_secret("AZURE_AI_ENDPOINT").strip())
        base_url = f"{parsed.scheme or 'https'}://{parsed.netloc or parsed.path}/openai/v1/"
        # No temperature / max_tokens: reasoning models (GPT-5.x, o-series) reject them
        return ChatOpenAI(base_url=base_url, api_key=get_secret("AZURE_AI_KEY"), model=model_id)

# --- Azure AI Search: index definition & clients ---
def build_index_definition():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=True,      # never returned in results (the app doesn't need it)
            stored=False,     # don't keep a second, retrievable copy of each vector
            vector_search_dimensions=EMBEDDING_DIM,
            vector_search_profile_name="vector-profile",
        ),
        SimpleField(name="role", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config", parameters=HnswParameters(metric="cosine"))],
        # Scalar quantization: vectors kept in the index as int8 instead of float32 (~4x smaller)
        compressions=[ScalarQuantizationCompression(
            compression_name="sq-int8",
            parameters=ScalarQuantizationParameters(quantized_data_type="int8"),
        )],
        profiles=[VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="hnsw-config",
            compression_name="sq-int8",
        )],
    )
    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

@st.cache_resource
def get_search_clients():
    """Connects to Azure AI Search and creates the index on first run."""
    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)
    try:
        index_client.get_index(INDEX_NAME)
    except ResourceNotFoundError:
        index_client.create_index(build_index_definition())
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)
    return index_client, search_client

def odata_str(value: str) -> str:
    """Escapes single quotes for OData filter strings."""
    return value.replace("'", "''")

def make_chunk_id(source: str, page: int, chunk_idx: int) -> str:
    """Deterministic ID: re-processing a page overwrites its chunks instead of duplicating them."""
    return hashlib.sha1(f"{source}|{page}|{chunk_idx}".encode("utf-8")).hexdigest()

def call_with_retry(fn, status_box=None, attempts=3, wait_s=10):
    """Retries a call on rate-limit / quota errors."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "quota" in msg or "rate" in msg) and attempt < attempts - 1:
                if status_box:
                    status_box.write(f"⏱️ Rate limit hit. Pausing {wait_s}s...")
                time.sleep(wait_s)
            else:
                raise

def search_manuals(query: str, role_filter: str, k: int = 4):
    """Hybrid search (keywords + vectors), restricted to the user's role."""
    vector = embeddings.embed_query(query)
    results = search_client.search(
        search_text=query,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        filter=role_filter,
        select=["content", "source", "page"],
        top=k,
    )
    return list(results)

def list_manuals(role_filter: str):
    """Returns (total_chunks, [(file_name, chunk_count), ...]) visible to this role."""
    results = search_client.search(
        search_text="*",
        filter=role_filter,
        facets=["source,count:1000"],
        include_total_count=True,
        top=0,
    )
    facets = results.get_facets() or {}
    total = results.get_count() or 0
    return total, [(f["value"], f["count"]) for f in facets.get("source", [])]

def get_last_processed_page(source: str) -> int:
    results = search_client.search(
        search_text="*",
        filter=f"source eq '{odata_str(source)}'",
        order_by=["page desc"],
        select=["page"],
        top=1,
    )
    for r in results:
        return int(r["page"])
    return 0

def delete_manual(source: str) -> int:
    results = search_client.search(search_text="*", filter=f"source eq '{odata_str(source)}'", select=["id"])
    ids = [{"id": r["id"]} for r in results]
    for i in range(0, len(ids), 1000):
        search_client.delete_documents(documents=ids[i:i + 1000])
    return len(ids)

def wipe_index():
    index_client.delete_index(INDEX_NAME)
    index_client.create_index(build_index_definition())

# Initialize Active Components
embeddings = get_embedding_engine()
vision_llm = get_vision_llm()
index_client, search_client = get_search_clients()
user_filter = ROLE_FILTERS[st.session_state['user_role']]

# --- Active Manuals Sidebar Status (role-filtered) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Active Manuals in DB")

try:
    total_count, manuals = list_manuals(user_filter)
    st.sidebar.caption(f"Total Chunks: {total_count}")
    if manuals:
        for file_name, count in manuals:
            st.sidebar.markdown(f"- 📄 `{file_name}` ({count})")
    else:
        st.sidebar.warning("Database is currently empty.")
except Exception as e:
    st.sidebar.caption(f"Could not read library status: {e}")
    manuals = []

# --- Admin Functions (technicians only) ---
if st.session_state['user_role'] == 'technician':
    st.sidebar.markdown("---")

    with st.sidebar.expander("Admin: Upload Manual", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        doc_role = st.selectbox("Assign Access Level", ["customer", "technician"])

        if uploaded_file and st.button("Process & Secure"):
            status_box = st.status("Initializing Heavy-Duty Processing...", expanded=True)
            current_page_num = 0

            try:
                # 1. Checkpoint: find the last page already stored for this file
                last_processed_page = 0
                try:
                    last_processed_page = get_last_processed_page(uploaded_file.name)
                except Exception as e:
                    status_box.write(f"⚠️ Checkpoint notice: Starting fresh. ({e})")

                doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                total_pages = len(doc)
                current_page_num = last_processed_page

                if last_processed_page >= total_pages:
                    status_box.update(label="✅ Document already fully processed!", state="complete")
                    st.stop()

                status_box.update(label=f"PDF Loaded. Resuming from page {last_processed_page + 1} of {total_pages}...", state="running")
                progress_bar = st.progress(last_processed_page / total_pages)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                # 2. Page-by-page streaming ingestion
                for i in range(last_processed_page, total_pages):
                    current_page_num = i + 1
                    page = doc.load_page(i)

                    needs_vision = False
                    for img in page.get_images():
                        base_image = doc.extract_image(img[0])
                        if base_image and base_image.get("width", 0) > 200 and base_image.get("height", 0) > 200:
                            needs_vision = True
                            break

                    if needs_vision:
                        status_box.write(f"📸 Page {current_page_num}/{total_pages}: Large diagram detected. Processing with Gemini Vision...")
                        pix = page.get_pixmap(dpi=72)
                        img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

                        prompt = """
                        You are a technical documentation assistant.
                        1. Extract all text from this manual page exactly as written.
                        2. If there are any screenshots, diagrams, tables, or UI panels, write a highly detailed description of them.
                        Format your response cleanly.
                        """
                        message = HumanMessage(content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        ])
                        page_content = call_with_retry(lambda: vision_llm.invoke([message]).content, status_box)
                    else:
                        status_box.write(f"📄 Page {current_page_num}/{total_pages}: Direct text extraction...")
                        page_content = page.get_text()

                    chunks = text_splitter.split_documents([Document(page_content=page_content or "")])
                    texts = [c.page_content for c in chunks if c.page_content.strip()]

                    if texts:
                        vectors = call_with_retry(lambda: embeddings.embed_documents(texts), status_box)
                        docs_to_upload = [
                            {
                                "id": make_chunk_id(uploaded_file.name, current_page_num, j),
                                "content": text,
                                "content_vector": vector,
                                "role": doc_role,
                                "source": uploaded_file.name,
                                "page": current_page_num,
                            }
                            for j, (text, vector) in enumerate(zip(texts, vectors))
                        ]
                        results = search_client.merge_or_upload_documents(documents=docs_to_upload)
                        failed = [r.key for r in results if not r.succeeded]
                        if failed:
                            raise Exception(f"{len(failed)} chunk(s) failed to upload on page {current_page_num}.")

                    progress_bar.progress(current_page_num / total_pages)

                status_box.update(label=f"✅ Processing completed: {uploaded_file.name}", state="complete", expanded=False)
                st.success(f"Storage complete for {uploaded_file.name}!")

            except Exception as e:
                status_box.update(label=f"❌ Stopped at Page {current_page_num}", state="error")
                st.error(f"Pipeline Error: {str(e)}")
                st.info("Your progress up to the last saved page is stored in Azure AI Search. Click 'Process & Secure' again to resume.")

    with st.sidebar.expander("Admin: Delete a Manual", expanded=False):
        if manuals:
            manual_to_delete = st.selectbox("Select manual", [m[0] for m in manuals])
            if st.button("🗑️ Delete this manual"):
                try:
                    deleted = delete_manual(manual_to_delete)
                    st.success(f"Deleted {deleted} chunks of {manual_to_delete}.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        else:
            st.caption("No manuals to delete.")

    with st.sidebar.expander("Admin: Danger Zone", expanded=False):
        if st.button("⚠️ Wipe Entire Database"):
            try:
                wipe_index()
                st.success("Database completely cleared!")
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")


# --- Agent Tools (calculation tools unchanged) ---

@tool
def calculate_itu_attenuations(lat: float, lon: float, distance_km: float, frequency_ghz: float, availability_pct: float) -> str:
    """
    Computes Free Space Loss (FSL), Atmospheric Attenuation, and Rain Attenuation for a specific location.
    Requires: latitude, longitude, link distance (km), frequency (GHz), and target availability (%).
    """
    f = frequency_ghz
    d = distance_km
    p = (100.0 - availability_pct)

    fsl_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 92.45

    el = 0.0
    rho = 7.5
    P_hpa = 1013.25
    T_kelvin = 288.15

    gamma_gas = itur.models.itu676.gaseous_attenuation_terrestrial_path(d, f, el, rho, P_hpa, T_kelvin, 'approx')

    tau = 0.0
    a_rain = itur.models.itu530.rain_attenuation(lat, lon, d, f, el, p, tau)

    return (f"FSL: {fsl_db:.2f} dB\n"
            f"Atmospheric Gas Attenuation: {gamma_gas.value:.2f} dB\n"
            f"Rain Attenuation ({availability_pct}% availability): {a_rain.value:.2f} dB")


@tool
def calculate_link_availability(qam_level: int, channel_bw_mhz: float, distance_km: float,
                                antenna_diameter_m: float, frequency_ghz: float, tx_power_dbm: float,
                                lat: float, lon: float) -> str:
    """
    Calculates expected link availability (%) by running a full link budget.
    Requires: QAM, BW (MHz), distance (km), antenna diameter (m), frequency (GHz), Tx Power (dBm), lat, and lon.
    """
    rx_thresh_str = calculate_rx_threshold.invoke({"qam_level": qam_level, "channel_bw_mhz": channel_bw_mhz, "frequency_ghz": frequency_ghz})
    rx_threshold = float(rx_thresh_str.split(":")[1].split("dBm")[0].strip())

    ant_specs_str = calculate_antenna_specs.invoke({"frequency_ghz": frequency_ghz, "diameter": antenna_diameter_m, "unit": "m"})
    gain_dbi = float(ant_specs_str.split("Gain:")[1].split("dBi")[0].strip())

    f = frequency_ghz
    d = distance_km
    el = 0.0

    fsl_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 92.45

    gamma_gas = itur.models.itu676.gaseous_attenuation_terrestrial_path(
        d, f, el, 7.5, 1013.25, 288.15, 'approx'
    ).value

    rx_clear_sky = tx_power_dbm + (2 * gain_dbi) - fsl_db - gamma_gas
    fade_margin = rx_clear_sky - rx_threshold

    if fade_margin <= 0:
        return f"Link fails in clear sky. Fade margin is {fade_margin:.2f} dB."

    def fade_difference(p_exceedance):
        a_rain = itur.models.itu530.rain_attenuation(lat, lon, d, f, el, p_exceedance, 0.0).value
        return a_rain - fade_margin

    try:
        result = root_scalar(fade_difference, bracket=[1e-5, 50], method='brentq')
        availability = 100.0 - result.root
        return (f"Clear Sky Rx Level: {rx_clear_sky:.2f} dBm\n"
                f"Fade Margin: {fade_margin:.2f} dB\n"
                f"Expected Availability: {availability:.5f}%")
    except ValueError:
        return f"Fade Margin is {fade_margin:.2f} dB. Availability > 99.9999% (exceeds standard ITU bounds)."


@tool
def calculate_antenna_specs(frequency_ghz: float, diameter: float, unit: str) -> str:
    """
    Calculates parabolic dish antenna gain (dBi) and Half-Power Beamwidth (degrees).
    Requires: frequency (GHz), diameter, and unit ('cm' or 'ft').
    """
    if unit.lower() == 'ft':
        diameter_m = diameter * 0.3048
    elif unit.lower() == 'cm':
        diameter_m = diameter / 100.0
    else:
        diameter_m = diameter

    c = 299792458.0
    f_hz = frequency_ghz * 1e9
    wavelength_m = c / f_hz
    efficiency = 0.60

    gain_linear = efficiency * ((math.pi * diameter_m) / wavelength_m)**2
    gain_dbi = 10 * math.log10(gain_linear)
    beamwidth_deg = 70 * (wavelength_m / diameter_m)

    return f"Antenna Gain: {gain_dbi:.2f} dBi, Half-Power Beamwidth: {beamwidth_deg:.2f}°"


@tool
def calculate_rx_threshold(qam_level: int, channel_bw_mhz: float, frequency_ghz: float) -> str:
    """
    Estimates the receive level threshold (dBm) based on the Shannon limit, coding rate (0.88),
    and dynamically scaled Noise Figure.
    Requires: QAM level (e.g., 1024), channel bandwidth (MHz), and frequency (GHz).
    """
    m = math.log2(qam_level)
    coding_rate = 0.88
    spectral_efficiency = m * coding_rate

    snr_linear = (2 ** spectral_efficiency) - 1
    snr_req_db = 10 * math.log10(snr_linear) + 1.6 + 4.0

    freq_bounded = max(6.0, min(84.0, frequency_ghz))
    nf_db = 3.0 + ((7.0 - 3.0) / (84.0 - 6.0)) * (freq_bounded - 6.0)

    thermal_noise_dbm = -174.0 + 10 * math.log10(channel_bw_mhz * 1e6)
    rx_threshold_dbm = thermal_noise_dbm + nf_db + snr_req_db

    return (f"Rx Threshold: {rx_threshold_dbm:.2f} dBm\n"
            f"(Calculated using {nf_db:.2f}dB NF and {snr_req_db:.2f}dB Required SNR)")


# --- Main Search & Telecom Chat Interface ---
st.title("💬 IP50EX/CX/GP/20N-Series Agent")
st.caption(f"Active Agent Reasoning Engine: **{selected_model_label}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a technical/telecom engineering question:")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner(f"Agent is analyzing your request using {selected_model_id}..."):
            try:
                chat_llm = get_chat_llm(selected_model_id)

                # Role filter is fixed here, outside the tool, so the LLM cannot change it
                def fetch_docs(search_query: str) -> str:
                    results = search_manuals(search_query, user_filter, k=4)
                    if not results:
                        return "No relevant information found within authorized manuals."
                    return "\n\n".join(
                        f"[Source: {r['source']}, page {r['page']}]\n{r['content']}" for r in results
                    )

                search_hardware_manuals = Tool(
                    name="search_hardware_manuals",
                    description="Searches authorized technical manuals. Use this strictly to answer questions about hardware specs, configurations, installation, and IP50/VX-Series documentation.",
                    func=fetch_docs,
                )

                tools = [
                    calculate_itu_attenuations,
                    calculate_antenna_specs,
                    calculate_rx_threshold,
                    calculate_link_availability,
                    search_hardware_manuals,
                ]

                system_prompt = SystemMessage(content="""You are an expert telecom and wireless hardware engineering assistant.
                You have tools available to you.
                - If the user asks for a calculation, strictly use the relevant calculation tool.
                - If the user asks for hardware specs or documentation, use the document search tool.
                - Do NOT guess hardware specs or math equations. Rely strictly on your tools.
                - When you answer from the manuals, mention the source document and page.

                At the very end of your final answer, provide exactly 2 highly relevant follow-up questions the user could ask. Format them as a bulleted list under the bold heading: **Suggested Follow-up Questions:**
                """)

                agent = create_react_agent(chat_llm, tools, prompt=system_prompt)

                # Error messages are shown in the UI but not sent back to the agent
                chat_history = [(m["role"], m["content"]) for m in st.session_state.messages if not m.get("is_error")]

                response = agent.invoke({"messages": chat_history})
                final_answer = response["messages"][-1].content
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                error_msg = f"❌ Agent Execution Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "is_error": True})
