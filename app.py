import os
import gradio as gr
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openrouter import OpenRouter


# NLP
NLP_MODEL_PATH = "model/best_phayathaibert_model"
try:
    nlp_pipeline = pipeline("text-classification", model=NLP_MODEL_PATH, tokenizer=NLP_MODEL_PATH)
except Exception as e:
    nlp_pipeline = None
    print(f"Warning: ไม่สามารถโหลดโมเดล NLP ได้: {e}")

#  โหลด RAG (Chroma DB & Embeddings)

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory="model/chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",           
        search_kwargs={"k": 3}
    )
except Exception as e:
    retriever = None
    print(f"Warning: ไม่สามารถโหลด RAG ได้: {e}")

SYSTEM_PROMPT = """Role:
    คุณเป็นผู้เชี่ยวชาญด้านความมั่นคงปลอดภัยไซเบอร์ และการตรวจสอบข้อเท็จจริงของข้อความ SMS ในประเทศไทย ที่มีประสบการณ์สูงมานานกว่า 50 ปี

Task:
    หน้าที่ของคุณคือวิเคราะห์ข้อความ SMS ที่ได้รับ เพื่อตัดสินอย่างตรงไปตรงมาและเป็นกลางว่าข้อความนี้เป็น "SMS หลอกลวง (Scam)" หรือ "SMS ปกติ (Normal)" 
    คุณต้องไม่อคติว่าข้อความที่ได้รับมาจะเป็น Scam เสมอไป แต่ให้ประเมินจากหลักฐานและบริบทที่มีอยู่จริง

Analysis method:
    1. วิเคราะห์ลักษณะของข้อความ: มองหาจุดที่น่าสงสัย (เช่น ลิงก์แปลกปลอม, การเร่งรัด, ข้อเสนอดีเกินจริง)
    2. ตรวจสอบความถูกต้องของลิงก์หรือโดเมน: เปรียบเทียบลิงก์ในข้อความกับโดเมนอย่างเป็นทางการขององค์กรนั้นๆ
    3. พิจารณาเจตนาของข้อความ: ข้อความมีเจตนาเพื่อหลอกให้คลิก/โอนเงิน หรือเพียงแค่ให้ข้อมูล/ทำการตลาดตามปกติ
    4. ใช้ข้อมูลจาก Context ที่เกี่ยวข้องเพื่อสนับสนุนการวิเคราะห์ของคุณ

Response format:
   ให้วิเคราะห์ออกมาแล้วตอบสั้นๆเป็นคำว่า "ปกติ" หรือ "แสกมเมอร์"เท่านั้น ห้ามตอบอย่างอื่นนอกจากคำว่า "ปกติ" หรือ "แสกมเมอร์"

Important:
    - ห้ามตีความเข้าข้างว่าเป็น Scam หากข้อความนั้นเป็นข้อความแจ้งเตือนทั่วไป (เช่น OTP, โปรโมชัน AIS/TRUE, แจ้งส่งพัสดุ) ที่ไม่มีลิงก์อันตรายหรือการแอบอ้าง
    - ให้ใช้เฉพาะข้อมูลจาก Context ที่ให้มาเท่านั้นในการอ้างอิงนโยบายที่ถูกต้อง ห้ามแต่งนโยบายหรือชื่อโดเมนขึ้นมาเองเด็ดขาด
    - หาก Context ไม่เพียงพอและข้อความไม่มีลักษณะอันตรายชัดเจน ให้วิเคราะห์จากจากลักษณะของข้อความแทน"""

USER_PROMPT_TEMPLATE = """ข้อความ SMS: {sms_message}
นโยบายอ้างอิง:
{retrieved_context}"""


# ฟังก์ชันสำหรับการวิเคราะห์
def analyze_scam_sms(sms_message: str):
   
    nlp_result = "ไม่สามารถใช้งานโมเดล NLP ได้"
    is_scam = True  
    
    if nlp_pipeline:
        try:
            preds = nlp_pipeline(sms_message)
            label = preds[0]['label']
            score = preds[0]['score']
            nlp_result = f"Label: {label} (ความมั่นใจ: {score:.4f})"
            
            # ตรวจสอบ Label ว่าเป็น Scam หรือ ปกติ
            if "0" in label or "normal" in label.lower() or "ปกติ" in label:
                is_scam = False
            else:
                is_scam = True
                
        except Exception as e:
            nlp_result = f"เกิดข้อผิดพลาด NLP: {str(e)}"
            is_scam = True # ถ้า NLP Error ให้ส่งไป LLM ต่อ
            
    if not is_scam:
        return nlp_result, "ข้อความปกติ"
            
    # ส่วนที่ 2: RAG + LLM Prediction (ทำงานเมื่อ NLP ตรวจพบว่าเป็น Scam)
    
    llm_result = "ไม่สามารถใช้งานโมเดล LLM + RAG ได้"
    if retriever:
        try:
            # Retrieve Context
            retrieved_docs = retriever.invoke(sms_message)
            context = "\n\n---\n\n".join([
                f"[ที่มา: {doc.metadata.get('source', 'ระบุแหล่งที่มา')}]\n{doc.page_content}"
                for doc in retrieved_docs
            ])
            
            user_prompt = USER_PROMPT_TEMPLATE.format(
                sms_message=sms_message,
                retrieved_context=context
            )
            
            # ดึง API Key จาก Secrets ของ HuggingFace Space 
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                llm_result = "กรุณาตั้งค่า OPENROUTER_API_KEY ใน Variables/Secrets ของ Hugging Face Space"
            else:
                with OpenRouter(api_key=api_key) as client:
                    response = client.chat.send(
                        model="qwen/qwen3-32b",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ]
                    )
                llm_result = response.choices[0].message.content
        except Exception as e:
            llm_result = f"เกิดข้อผิดพลาด LLM: {str(e)}"
            
    return nlp_result, llm_result


# Gradio UI สำหรับ HuggingFace Space

with gr.Blocks(title="SMS Scam Detection") as demo:
    gr.Markdown("# 🛡️ ระบบตรวจจับ SMS หลอกลวง (NLP + RAG + LLM)")
    gr.Markdown("พิมพ์หรือวางข้อความ SMS ที่ต้องการตรวจสอบ ระบบจะวิเคราะห์ด้วย 2 โมเดล: **PhayathaiBERT (NLP)** และ **Qwen 3 32b (LLM + RAG)**")
    
    with gr.Row():
        with gr.Column():
            sms_input = gr.Textbox(label="ข้อความ SMS", placeholder="วางข้อความ SMS ที่นี่...", lines=5)
            analyze_btn = gr.Button("วิเคราะห์ข้อความ", variant="primary")
            
        with gr.Column():
            nlp_output = gr.Textbox(label="ผลลัพธ์จาก NLP (PhayathaiBERT)", lines=2, interactive=False)
            llm_output = gr.Textbox(label="ผลลัพธ์จาก RAG + LLM (Qwen 3 32b)", lines=5, interactive=False)
            
    analyze_btn.click(
        fn=analyze_scam_sms, 
        inputs=sms_input, 
        outputs=[nlp_output, llm_output]
    )

if __name__ == "__main__":
    demo.launch()
