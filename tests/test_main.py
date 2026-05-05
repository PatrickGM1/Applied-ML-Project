import unittest
from fastapi.testclient import TestClient
from main import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_stub(self):
        response = self.client.post("/predict", json={"text": "Some news article text"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("label", data)
        self.assertIn("message", data)


if __name__ == '__main__':
    unittest.main()

