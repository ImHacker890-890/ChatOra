import requests

class GitHubAPI:
    def __init__(self, token):
        self.token = token
        self.headers = {"Authorization": f"token {token}"}
    
    def get_code_samples(self, repo_url):
        response = requests.get(
            f"https://api.github.com/repos/{repo_url}/contents",
            headers=self.headers
        )
        return [f for f in response.json() if f['name'].endswith('.py')]
