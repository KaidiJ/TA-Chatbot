from piazza_api import Piazza
import sqlite3, os
from bs4 import BeautifulSoup

def fetch_piazza_posts():
    p = Piazza()
    p.user_login(email="", password="")
    course = p.network("lr557t7zde528n")

    posts_data = []
    for post in course.iter_all_posts(limit=20, sleep=0.2):
        post_info = {
            'uid': post['history'][0]['uid'] if 'uid' in post['history'][0] else 'Unknown',
            'type': post['type'] if 'type' in post else 'Unknown',
            'content': post['history'][0]['content'] if 'content' in post['history'][0] else 'No content',
            'subject': post['history'][0]['subject'] if 'subject' in post['history'][0] else 'No subject',
            'followups': []
        }

        if 'children' in post:
            for child in post['children']:
                if post_info['type'] == 'question':
                    if child['type'] == 'i_answer':
                        post_info.setdefault('answers', []).append(
                            {'type': 'instructor', 'content': child['history'][0]['content']})
                    elif child['type'] == 's_answer':
                        post_info.setdefault('answers', []).append(
                            {'type': 'student', 'content': child['history'][0]['content']})

                if child['type'] == 'followup':
                    followup_info = {
                        'content': child['subject'],
                        'replies': []
                    }
                    for sub_child in child['children']:
                        reply_type = 'instructor' if sub_child['type'] == 'i_answer' else 'student'
                        reply_info = {
                            'type': reply_type,
                            'content': sub_child.get('subject', '') + ' ' + sub_child.get('content', ''),
                        }
                        followup_info['replies'].append(reply_info)

                    post_info['followups'].append(followup_info)

        posts_data.append(post_info)

    return posts_data

#Data pre for HTML content
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text

def post_info_to_text(post_info):
    text_parts = []
    if 'subject' in post_info:
        text_parts.append(clean_html(post_info['subject']))
    if 'content' in post_info:
        text_parts.append(clean_html(post_info['content']))
    if 'answers' in post_info:
        for answer in post_info['answers']:
            answer_text = f"{answer['type']} answer: {clean_html(answer['content'])}"
            text_parts.append(answer_text)
    if 'followups' in post_info:
        for followup in post_info['followups']:
            followup_text = clean_html(followup['content'])
            text_parts.append(f"followup: {followup_text}")
            if 'replies' in followup:
                for reply in followup['replies']:
                    reply_text = f"{reply['type']} reply: {clean_html(reply['content'])}"
                    text_parts.append(reply_text)
    return " ".join(text_parts)

def save_post_to_db(post_info, db_path="piazza_posts.db"):
    post_text = post_info_to_text(post_info)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')

    try:
        c.execute("INSERT INTO posts (content) VALUES (?)", (post_text,))
        print("Saved successfully.")
    except sqlite3.IntegrityError as e:
        print(f"Error saving post: {e}")

    conn.commit()
    conn.close()


if os.path.exists("piazza_posts.db"):
    os.remove("piazza_posts.db")

piazza_posts_data = fetch_piazza_posts()
for post_info in piazza_posts_data:
    save_post_to_db(post_info)

# def read_and_combine_posts_content(db_path="piazza_posts.db"):
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()
#     c.execute("SELECT content FROM posts")
#     all_posts_content = c.fetchall()
#     combined_content = " ".join(post_content[0] for post_content in all_posts_content)
#     conn.close()
#     return combined_content

# combined_post_text = read_and_combine_posts_content()
# print(combined_post_text)
