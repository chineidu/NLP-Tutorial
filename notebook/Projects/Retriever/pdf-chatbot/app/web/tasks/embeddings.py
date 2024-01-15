from app.chat import create_embeddings_for_pdf
from app.web.db.models import Pdf
from app.web.files import download
from celery import shared_task


@shared_task()
def process_document(pdf_id: int):
    pdf = Pdf.find_by(id=pdf_id)
    with download(pdf.id) as pdf_path:
        create_embeddings_for_pdf(pdf.id, pdf_path)
