import unittest
import rest_api


class ApiTest(unittest.TestCase):

    def setUp(self) -> None:
        self.api = rest_api.Api(base_url="http://homeassistant:8123/api/", api="states/sensor.person_detector")
        self.api.header = ("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJmYzJlZWI1N2ZmMDE0MDc5YmE4OGE1NDQ0NWU"
                           "4NzlkYiIsImlhdCI6MTYzMzg0NjAzMCwiZXhwIjoxOTQ5MjA2MDMwfQ.hgpZZM8l90WZjAXMAWm6RImPQVyN"
                           "YnkwRV4pBvklTv4")
        self.body = {
            "state": "clear",
            "attributes": {
                "detected": "person",
                "location": "backroom"
            }
        }

    def tearDown(self) -> None:
        pass

    def test_post(self):
        self.api.post(data=self.body)


if __name__ == '__main__':
    unittest.main()
