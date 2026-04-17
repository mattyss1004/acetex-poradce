"""
app.py
------
Streamlit chat UI for the Acetex RAG chatbot with password protection.

Run with:
    streamlit run app.py
or use the provided start.sh script.
"""

import streamlit as st
from rag_chain import answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Acetex Poradce",
    page_icon="☀️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Password Protection ───────────────────────────────────────────────────────
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "AcetexDemo2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("### ☀️ Acetex Poradce — Přihlášení")
        st.text_input(
            "Zadejte heslo pro přístup k demo verzi:", type="password", on_change=password_entered, key="password"
        )
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Nesprávné heslo. Zkuste to prosím znovu.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.markdown("### ☀️ Acetex Poradce — Přihlášení")
        st.text_input(
            "Zadejte heslo pro přístup k demo verzi:", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Nesprávné heslo. Zkuste to prosím znovu.")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f7f9fc; }

    /* Chat message bubbles */
    .user-bubble {
        background: #0066cc;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 6px 0 6px 15%;
        font-size: 0.97rem;
        line-height: 1.5;
    }
    .bot-bubble {
        background: #ffffff;
        color: #1a1a2e;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 6px 15% 6px 0;
        font-size: 0.97rem;
        line-height: 1.6;
        border: 1px solid #e0e6ef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .source-tag {
        display: inline-block;
        background: #eef3fb;
        color: #0055aa;
        border-radius: 10px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 3px 3px 0 0;
        text-decoration: none;
    }
    .source-tag:hover { background: #d6e4f7; }
    .thinking-label {
        color: #888;
        font-size: 0.85rem;
        font-style: italic;
        margin-left: 4px;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Input area */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1.5px solid #c8d8ee;
        padding: 10px 18px;
        font-size: 0.97rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0066cc;
        box-shadow: 0 0 0 2px rgba(0,102,204,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Acetex Poradce")
    st.markdown(
        "Váš průvodce světem fotovoltaiky, tepelných čerpadel a dotací. "
        "Ptejte se na cokoliv — odpovídám z dokumentace a webu Acetex."
    )
    st.divider()
    st.markdown("**Kontakt na Acetex**")
    st.markdown("📞 770 110 011")
    st.markdown("✉️ info@acetex.cz")
    st.markdown("🌐 [acetex.cz](https://acetex.cz)")
    st.divider()
    if st.button("🗑️ Vymazat konverzaci", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        "<div style='font-size:0.75rem;color:#aaa;margin-top:12px;'>"
        "Odpovědi vycházejí z dokumentace a webu Acetex. "
        "Pro aktuální ceny kontaktujte obchodní tým."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("### Acetex — Poradce pro fotovoltaiku a energetiku")
st.markdown(
    "<div style='color:#666;font-size:0.9rem;margin-bottom:16px;'>"
    "Zeptejte se na produkty, dotace, instalaci nebo technické detaily."
    "</div>",
    unsafe_allow_html=True,
)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show sources as small tags if available
        if msg.get("sources"):
            seen = set()
            tags = []
            for s in msg["sources"]:
                url = s.get("source", "")
                label = s.get("title", url)
                # Clean up label
                label = label.replace(" | Acetex", "").strip()
                if not label:
                    label = url.split("/")[-1] or "Zdroj"
                if url and url not in seen:
                    seen.add(url)
                    tags.append(
                        f'<a class="source-tag" href="{url}" target="_blank">'
                        f'📄 {label[:45]}</a>'
                    )
                elif not url and label not in seen:
                    seen.add(label)
                    tags.append(
                        f'<span class="source-tag">📄 {label[:45]}</span>'
                    )
            if tags:
                st.markdown(
                    "<div style='margin: 2px 0 10px 0;'>" + "".join(tags) + "</div>",
                    unsafe_allow_html=True,
                )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="question",
            placeholder="Napište svůj dotaz… (např. Jaká je dotace na fotovoltaiku?)",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Odeslat", use_container_width=True)

# ── Process input ─────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # Show thinking indicator and get answer
    with st.spinner("Hledám odpověď…"):
        result = answer(question)

    # Add bot response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

    # Rerun to refresh the chat display
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:40px 20px;color:#999;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>☀️</div>
        <div style='font-size:1rem;'>Začněte tím, že se zeptáte na cokoliv o fotovoltaice,<br>
        tepelných čerpadlech, bateriích nebo dotacích.</div>
        <div style='margin-top:20px;font-size:0.85rem;color:#bbb;'>
            Například: <em>Jak funguje NZÚ Light?</em> &nbsp;·&nbsp;
            <em>Jaká je záruka na Wattsonic?</em> &nbsp;·&nbsp;
            <em>Jak zapojit baterie GoodWe?</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
