import contextlib

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"

# --- UI translations ---

UI_TEXT = {
    "en": {
        "page_title": "BanglaLaw — Legal Information Assistant",
        "header_title": "⚖️ BanglaLaw",
        "header_subtitle": "AI-Powered Legal Information Assistant for Bangladesh",
        "chat_placeholder": "Type your legal question...",
        "searching": "Searching legal documents...",
        "low_confidence": "⚠️ This answer has low confidence.",
        "sources": "Sources",
        "view": "View",
        "accuracy": "Accuracy",
        "categories_title": "📂 Law Categories",
        "cat_land": "🏠 Land & Property Law",
        "cat_family": "👨‍👩‍👧 Family Law",
        "cat_labor": "👷 Labor Law",
        "cat_consumer": "🛒 Consumer Rights",
        "cat_criminal": "⚖️ Criminal Law",
        "cat_tenant": "🏢 Tenant Law",
        "cat_query_suffix": "Tell me about",
        "bookmarks_title": "🔖 Saved Answers",
        "no_bookmarks": "No saved answers yet.",
        "save": "🔖 Save",
        "saved_toast": "Saved!",
        "thanks_toast": "Thanks!",
        "disclaimer": "⚠️ This is not legal advice. Consult a lawyer for your specific situation.",
        "api_error": "Cannot connect to backend API. Make sure the API server is running.",
        "rate_limit": "Please try again in a moment. (Rate limit)",
        "toggle_lang": "বাংলা",
        "lawyer_title": "👨‍⚖️ Need a Lawyer?",
        "lawyer_info": "🔗 [National Legal Aid Services](http://www.nlaso.gov.bd/)\n\n🔗 [Bangladesh Bar Council](http://www.barcouncil.gov.bd/)",
    },
    "bn": {
        "page_title": "BanglaLaw — আইনি তথ্য সহকারী",
        "header_title": "⚖️ BanglaLaw",
        "header_subtitle": "বাংলাদেশের আইন বিষয়ক AI তথ্য সহকারী",
        "chat_placeholder": "আপনার আইনি প্রশ্ন লিখুন...",
        "searching": "আইনি তথ্য খোঁজা হচ্ছে...",
        "low_confidence": "⚠️ এই উত্তরের নির্ভুলতা কম।",
        "sources": "সূত্র",
        "view": "দেখুন",
        "accuracy": "নির্ভুলতা",
        "categories_title": "📂 আইনের বিভাগ",
        "cat_land": "🏠 ভূমি ও সম্পত্তি আইন",
        "cat_family": "👨‍👩‍👧 পারিবারিক আইন",
        "cat_labor": "👷 শ্রম আইন",
        "cat_consumer": "🛒 ভোক্তা অধিকার",
        "cat_criminal": "⚖️ ফৌজদারি আইন",
        "cat_tenant": "🏢 ভাড়াটিয়া আইন",
        "cat_query_suffix": "সম্পর্কে জানতে চাই",
        "bookmarks_title": "🔖 সংরক্ষিত উত্তর",
        "no_bookmarks": "কোনো উত্তর সংরক্ষিত হয়নি।",
        "save": "🔖 সংরক্ষণ",
        "saved_toast": "সংরক্ষিত!",
        "thanks_toast": "ধন্যবাদ!",
        "disclaimer": "⚠️ এটি আইনি পরামর্শ নয়। নির্দিষ্ট পরিস্থিতির জন্য আইনজীবীর সাথে পরামর্শ করুন।",
        "api_error": "Backend API তে সংযোগ করা যাচ্ছে না। API server চালু আছে কিনা নিশ্চিত করুন।",
        "rate_limit": "অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন। (Rate limit)",
        "toggle_lang": "English",
        "lawyer_title": "👨‍⚖️ আইনজীবী প্রয়োজন?",
        "lawyer_info": "🔗 [জাতীয় আইনগত সহায়তা সংস্থা](http://www.nlaso.gov.bd/)\n\n🔗 [বাংলাদেশ বার কাউন্সিল](http://www.barcouncil.gov.bd/)",
    },
}

# --- Custom CSS ---

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

.bangla-text {
    font-family: 'Noto Sans Bengali', sans-serif;
    line-height: 1.8;
}
.english-text {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.header-container {
    text-align: center;
    padding: 1rem 0;
}
.header-container h1 {
    margin-bottom: 0.2rem;
}
.header-container p {
    color: gray;
    margin-top: 0;
}
.disclaimer-box {
    font-size: 0.8rem;
    color: gray;
    line-height: 1.5;
}
</style>
"""


def t(key: str) -> str:
    """Get translated UI text for current language."""
    lang = st.session_state.get("ui_lang", "en")
    return UI_TEXT[lang].get(key, key)


def get_font_class(language: str) -> str:
    """Return CSS class based on response language."""
    return "bangla-text" if language == "bn" else "english-text"


# --- Session state ---


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "en"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


# --- API calls ---


def query_api(question: str) -> dict | None:
    """Send question to the FastAPI backend."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error(t("api_error"))
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            st.warning(t("rate_limit"))
        else:
            st.error(f"Error: {e.response.status_code}")
        return None


def send_feedback(query: str, helpful: bool):
    """Send feedback to the API."""
    with contextlib.suppress(httpx.HTTPError):
        httpx.post(
            f"{API_BASE_URL}/feedback",
            json={"query": query, "helpful": helpful},
            timeout=10.0,
        )


# --- UI components ---


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Language toggle in top-right
    col_spacer, col_toggle = st.columns([5, 1])
    with col_toggle:
        if st.button(t("toggle_lang"), use_container_width=True):
            st.session_state.ui_lang = "bn" if st.session_state.ui_lang == "en" else "en"
            st.rerun()

    font_class = get_font_class(st.session_state.ui_lang)
    st.markdown(
        f"""
        <div class="header-container {font_class}">
            <h1>{t("header_title")}</h1>
            <p>{t("header_subtitle")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer(answer: str, language: str):
    """Render answer with appropriate font."""
    font_class = get_font_class(language)
    st.markdown(f'<div class="{font_class}">{answer}</div>', unsafe_allow_html=True)


def render_citations(citations: list[dict]):
    """Render source citations in an expander."""
    if not citations:
        return
    with st.expander(f"📚 {t('sources')} ({len(citations)})", expanded=False):
        for i, cite in enumerate(citations, 1):
            confidence_pct = round(cite["confidence"] * 100)
            st.markdown(
                f"**{i}.** {cite['source']}  \n"
                f"🔗 [{t('view')}]({cite['url']}) | "
                f"{t('accuracy')}: {confidence_pct}%"
            )


def render_chat():
    """Main chat interface."""
    # Handle pending sidebar query
    if "pending_query" in st.session_state and st.session_state.pending_query:
        question = st.session_state.pending_query
        st.session_state.pending_query = None

        with st.chat_message("assistant"):
            with st.spinner(t("searching")):
                result = query_api(question)

            if result:
                resp_lang = result.get("language", "en")
                render_answer(result["answer"], resp_lang)
                render_citations(result.get("citations", []))

                if not result["is_confident"]:
                    st.warning(t("low_confidence"))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "language": resp_lang,
                        "citations": result.get("citations", []),
                        "confidence": result.get("confidence", 0),
                        "query": question,
                    }
                )

    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer(msg["content"], msg.get("language", "en"))
                if "citations" in msg:
                    render_citations(msg["citations"])

                if msg.get("citations"):
                    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
                    with col1:
                        if st.button("👍", key=f"hist_up_{idx}"):
                            send_feedback(msg.get("query", ""), helpful=True)
                            st.toast(t("thanks_toast"))
                    with col2:
                        if st.button("👎", key=f"hist_down_{idx}"):
                            send_feedback(msg.get("query", ""), helpful=False)
                            st.toast(t("thanks_toast"))
                    with col3:
                        if st.button(t("save"), key=f"hist_bm_{idx}"):
                            st.session_state.bookmarks.append(
                                {
                                    "question": msg.get("query", ""),
                                    "answer": msg["content"],
                                    "citations": msg.get("citations", []),
                                }
                            )
                            st.toast(t("saved_toast"))
                    with col4:
                        if st.button("📱 WhatsApp", key=f"hist_wa_{idx}"):
                            import urllib.parse

                            share_text = msg["content"][:200]
                            wa_url = f"https://wa.me/?text={urllib.parse.quote(share_text)}"
                            st.markdown(
                                f'<meta http-equiv="refresh" content="0;url={wa_url}">',
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input(t("chat_placeholder")):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(t("searching")):
                result = query_api(question)

            if result:
                resp_lang = result.get("language", "en")
                render_answer(result["answer"], resp_lang)
                render_citations(result.get("citations", []))

                if not result["is_confident"]:
                    st.warning(t("low_confidence"))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "language": resp_lang,
                        "citations": result.get("citations", []),
                        "confidence": result.get("confidence", 0),
                        "query": question,
                    }
                )


def render_sidebar():
    with st.sidebar:
        font_class = get_font_class(st.session_state.ui_lang)

        st.markdown(f'<div class="{font_class}">', unsafe_allow_html=True)
        st.markdown(f"### {t('categories_title')}")

        category_keys = [
            "cat_land",
            "cat_family",
            "cat_labor",
            "cat_consumer",
            "cat_criminal",
            "cat_tenant",
        ]
        for key in category_keys:
            label = t(key)
            if st.button(label, use_container_width=True, key=f"sidebar_{key}"):
                category_name = label.split(" ", 1)[1]
                query = f"{t('cat_query_suffix')} {category_name}"
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.pending_query = query
                st.rerun()

        st.markdown("---")
        st.markdown(f"### {t('bookmarks_title')}")
        if st.session_state.bookmarks:
            for _i, bm in enumerate(st.session_state.bookmarks):
                with st.expander(f"{bm['question'][:40]}..."):
                    st.markdown(bm["answer"][:300] + "...")
        else:
            st.caption(t("no_bookmarks"))

        st.markdown("---")
        st.markdown(f"### {t('lawyer_title')}")
        st.markdown(t("lawyer_info"))
        st.markdown("---")
        st.markdown(f'<div class="disclaimer-box">{t("disclaimer")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    init_session_state()
    st.set_page_config(
        page_title=t("page_title"),
        page_icon="⚖️",
        layout="centered",
    )
    render_header()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
