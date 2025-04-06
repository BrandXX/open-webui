from datetime import datetime
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def get_current_date(self) -> str:
        """
        Get the current date.
        :return: The current date as a string.
        """
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {current_date}"

    def get_current_time(self) -> str:
        """
        Get the current time.
        :return: The current time as a string.
        """
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"Current Time: {current_time}"


# Usage
# tools = Tools()
# print("Today's date:", tools.get_current_date())
# print("Current Time:", tools.get_current_time())