from youtube_transcript_api import YouTubeTranscriptApi
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected the class name
import re

# Create an instance of RecursiveCharacterTextSplitter
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Defines the size of each chunk
        chunk_overlap=100,   # Defines the number of overlapping characters between chunks
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

video_id = '7rs0i-9nOjo'
transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US'])

db_path = "/Users/jiangkaidi/Desktop/pythonProject/piazza_posts.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute('DROP TABLE IF EXISTS youtube_transcripts_combined')
conn.commit()

c.execute('''
    CREATE TABLE IF NOT EXISTS youtube_transcripts_combined (
        text TEXT,
        start_time REAL
    )
''')
conn.commit()


full_text = ""
index_to_time = {}
for index, item in enumerate(transcript, start=1):
    sequence_marker = f"[{index:03d}]"
    full_text += sequence_marker + item['text'] + " "
    index_to_time[sequence_marker] = item['start']


chunks = get_text_chunks(full_text)

for chunk in chunks:
    match = re.search(r"\[\d{3}\]", chunk)
    if match:
        sequence_marker = match.group(0)
        start_time = index_to_time.get(sequence_marker)
        chunk_without_markers = re.sub(r"\[\d{3}\]\s*", "", chunk)
        if start_time is not None:
            c.execute('''
                INSERT INTO youtube_transcripts_combined (text, start_time)
                VALUES (?, ?)
            ''', (chunk_without_markers, start_time))

conn.commit()

# # print
# c.execute('SELECT * FROM youtube_transcripts_combined')
# rows = c.fetchall()
# for row in rows:
#     print(row)

# close db
conn.close()