from langchain_helper import update_vector_db
import schedule
import time


def job():
    print("Updating vector database...")
    update_vector_db()


schedule.every(1).minutes.do(job)


while True:
    schedule.run_pending()
    time.sleep(1)
