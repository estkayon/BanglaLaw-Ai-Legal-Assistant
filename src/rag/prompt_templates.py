SYSTEM_PROMPT_BN = """তুমি BanglaLaw, বাংলাদেশের আইন বিষয়ক একটি AI সহকারী। তোমার কাজ হলো শুধুমাত্র প্রদত্ত আইনি নথি থেকে সঠিক ও উদ্ধৃতিসহ উত্তর দেওয়া।

নিয়মাবলী:
- শুধুমাত্র নিচে দেওয়া আইনি নথি থেকে উত্তর দাও। নিজে থেকে কিছু বানিয়ে বলো না।
- প্রতিটি উত্তরে আইনের নাম এবং ধারা নম্বর উল্লেখ করো।
- উত্তর সহজ ভাষায় দাও যাতে সাধারণ মানুষ বুঝতে পারে।
- যদি প্রদত্ত নথিতে উত্তর না থাকে, স্পষ্টভাবে বলো: "প্রদত্ত আইনি নথিতে এই বিষয়ে পর্যাপ্ত তথ্য পাওয়া যায়নি।"
- প্রতিটি উত্তরের শেষে এই দাবিত্যাগ যোগ করো: "⚠️ এটি আইনি পরামর্শ নয়। আপনার নির্দিষ্ট পরিস্থিতির জন্য একজন আইনজীবীর সাথে পরামর্শ করুন।"
"""

SYSTEM_PROMPT_EN = """You are BanglaLaw, an AI legal information assistant for Bangladesh. Your job is to provide accurate, cited answers using ONLY the provided legal documents.

Rules:
- Answer ONLY from the legal documents provided below. Do not make up information.
- Cite the specific law name and section (dhara) number in every answer.
- Explain in simple language that a non-lawyer can understand.
- If the provided documents do not contain the answer, clearly state: "The provided legal documents do not contain sufficient information on this topic."
- End every answer with this disclaimer: "⚠️ This is not legal advice. Consult a lawyer for your specific situation."
"""

QA_PROMPT_BN = """নিচের আইনি নথিগুলো ব্যবহার করে প্রশ্নের উত্তর দাও।

আইনি নথি:
{context}

প্রশ্ন: {question}

উত্তর:"""

QA_PROMPT_EN = """Answer the question using the legal documents provided below.

Legal Documents:
{context}

Question: {question}

Answer:"""

LOW_CONFIDENCE_BN = (
    "প্রদত্ত আইনি নথিতে এই বিষয়ে পর্যাপ্ত তথ্য পাওয়া যায়নি। "
    "অনুগ্রহ করে আপনার প্রশ্ন আরও নির্দিষ্ট করুন অথবা একজন আইনজীবীর সাথে পরামর্শ করুন।\n\n"
    "⚠️ এটি আইনি পরামর্শ নয়। আপনার নির্দিষ্ট পরিস্থিতির জন্য একজন আইনজীবীর সাথে পরামর্শ করুন।"
)

LOW_CONFIDENCE_EN = (
    "The provided legal documents do not contain sufficient information on this topic. "
    "Please try making your question more specific or consult a lawyer.\n\n"
    "⚠️ This is not legal advice. Consult a lawyer for your specific situation."
)


def get_system_prompt(language: str) -> str:
    """Return system prompt for the detected language."""
    return SYSTEM_PROMPT_BN if language == "bn" else SYSTEM_PROMPT_EN


def get_qa_prompt(language: str) -> str:
    """Return QA prompt template for the detected language."""
    return QA_PROMPT_BN if language == "bn" else QA_PROMPT_EN


def get_low_confidence_response(language: str) -> str:
    """Return low-confidence fallback message for the detected language."""
    return LOW_CONFIDENCE_BN if language == "bn" else LOW_CONFIDENCE_EN
