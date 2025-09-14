from locust import HttpUser, task, between, LoadTestShape
import random

# Example user queries for the chatbot
messages = [
    "How do I pay for my phone?",
    "My phone is locked, what should I do?"
]

class RasaUser(HttpUser):
    # Each simulated user waits 1–3 seconds between requests
    wait_time = between(1, 3)

    @task
    def chat_with_bot(self):
        """Simulates a single user sending a message to the Rasa REST webhook"""
        user_message = random.choice(messages)
        payload = {"sender": f"user_{random.randint(1, 100000)}", "message": user_message}

        # Send POST request to Rasa webhook
        with self.client.post(
            "/webhooks/rest/webhook",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Error {response.status_code}: {response.text}")
            else:
                response.success()

# Custom load shape
class StepLoadShape(LoadTestShape):
    """
    Ramp up 10 users/sec until 500 users
    Then maintain ~20 requests/sec
    """

    target_users = 500
    spawn_rate = 10   # users per second
    hold_time = 300   # how long to hold test at peak (seconds)

    def tick(self):
        run_time = self.get_run_time()

        # Ramp up phase
        if run_time < self.target_users / self.spawn_rate:
            current_users = int(run_time * self.spawn_rate)
            return (current_users, self.spawn_rate)

        # Hold phase
        elif run_time < (self.target_users / self.spawn_rate) + self.hold_time:
            return (self.target_users, self.spawn_rate)

        # Stop test
        return None
