import requests

# Google Drive shared link
SHARED_LINK = "https://drive.google.com/file/d/1gQfPGhm9UCuzmaAO9zmgq1S3wmEcOr4f/view?usp=sharing"

def get_token():
    global SHARED_LINK
    def get_google_drive_file_id(drive_url):
        file_id = drive_url.split('/d/')[1].split('/')[0]
        return file_id

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def get_response_content(response):
        CHUNK_SIZE = 32768
        content = ""
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                content += chunk.decode('utf-8')
        return content

    file_id = get_google_drive_file_id(SHARED_LINK)
    base_url = "https://drive.google.com/uc?export=download&id="
    session = requests.Session()
    response = session.get(base_url + file_id, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(base_url + file_id, params=params, stream=True)

    return get_response_content(response)
