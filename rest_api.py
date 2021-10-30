import requests as rq


class Api:
    def __init__(self, **kwargs):
        self._base_url = kwargs.get("base_url", None)
        self._api = kwargs.get("api", None)
        self._concat_url = self._api
        self._headers = ""

    @property
    def _concat_url(self):
        return self._url

    @_concat_url.setter
    def _concat_url(self, api: str):
        self._url = self._base_url + api

    @property
    def header(self):
        return self._headers

    @header.setter
    def header(self, token: str):
        _token = token
        self._headers = {
            "Authorization": "Bearer " + _token,
            "content-type": "application/json",
        }

    def post(self, **kwargs):
        _body = kwargs.get("data", None)

        response = rq.post(self._concat_url, headers=self.header, json=_body)
        print(response.text)
        print(response)
