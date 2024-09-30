from typing import Dict, List
import pandas as pd
from enum import Enum
from Levenshtein import distance
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json

# Debug flag to enable/disable debug output
DEBUG = True

# Load the  logistic regression model BEST
with open("classifiers/logistic_regression_deduped.pkl", "rb") as model_file:
    model = pickle.load(model_file)
# Load the  random forest model BEST
# with open("classifiers/random_forest_deduped.pkl", "rb") as model_file:
# model = pickle.load(model_file)

# Load the utterances used for the chatbot
with open("utterances.json") as utterances_file:
    utterances = json.load(utterances_file)

# Load the count vectorizer used during training to ensure the input format matches
with open("classifiers/count_vectorizer.pkl", "rb") as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

# Load the restaurant database
db = pd.read_csv("restaurant_info.csv")

# The storage object to keep track of the conversation state
storage = {
    "preferences": {},
    "suggestions": {},
    "current_suggestion": {},
    "exclude": [],
}

# The possible states of the dialog
class DialogState(Enum):
    WELCOME = 1  # starting state
    ASK_PREFERENCES = 2  # User provides preferences
    ASK_CLEAR_PREFERENCES = 3  # Ask for more clear answers
    NO_RESTAURANT_FOUND = 4  # 404 No restaurant found
    SUGGEST_RESTAURANT = 5  # Suggest a restaurant somehow
    DELETE_DISLIKED_RESTAURANT = 6  # User dislikes the suggestion. delete it
    OTHER_REQUEST = 7  # Other user request
    GOODBYE = 8  # End

# The possible dialog acts
class DialogAct(Enum):
    ACK = "ack" # Acknowledge
    AFFIRM = "affirm" # Positive confirmation
    BYE = "bye" # Greeting at the end of the dialog
    CONFIRM = "confirm" # Check if given information confirms to query
    DENY = "deny" # Reject system suggestion
    HELLO = "hello" # Greeting at the start of the dialog
    INFORM = "inform" # State a preference or other information
    NEGATE = "negate" # Negation
    NULL = "null" # Noise or utterance without content
    REPEAT = "repeat" # Ask for repetition
    REQALTS = "reqalts" # Request alternative suggestions
    REQMORE = "reqmore" # Request more suggestions
    REQUEST = "request" # Ask for information
    RESTART = "restart" # Attempt to restart the dialog
    THANKYOU = "thankyou" # Axpress thanks

# The possible preferences for each category
options = {
    "food": db["food"].dropna().unique(),
    "pricerange": db["pricerange"].dropna().unique(),
    "area": db["area"].dropna().unique(),
}

# The keywords used to extract preferences from the user input
keywords = {
    "food": ["food", "cuisine"],
    "pricerange": ["price", "cost", "prices", "priced"],
    "area": ["location", "area", "part"],
}

# Helper function to reply with a formatted utterance
def reply(name, **kwargs):
    print(utterances[name].format(**kwargs))

# Helper function to update the storage
def update_storage(table: str, dict: Dict):
    if table not in storage:
        storage[table] = {}

    for key, value in dict.items():
        if value:
            storage[table][key] = value

# Extract the preferences from the user input
def extract_preferences(user_input):
    splitted_input = user_input.split()
    preferences = {"food": None, "pricerange": None, "area": None}
    suggestions = {"food": None, "pricerange": None, "area": None}
    used = []

    # loop through the keywords and values
    for key, values in keywords.items():
        min_distance = 1000
        words = []

        for i, value in enumerate(splitted_input):
            # check if the value is in the options
            if value in options[key]:
                preferences[key] = value
                used.append(i)

            # check if the value is in the list of keywords
            elif value in values and splitted_input[i - 1] in options[key]:
                preferences[key] = splitted_input[i - 1]

            # The value is not in the options, but it is close to one of the options
            # Calculate the distance between the value and the options
            elif value in values and i - 1 not in used:
                for option in options[key]:
                    dist = distance(splitted_input[i - 1], option)
                    if dist < min_distance:
                        min_distance = dist
                        suggestions[key] = option

    return (preferences, suggestions)

# Get more information about the restaurant
def get_more_info(restaurant, user_input):
    if "address" in user_input:
        print(f"Address: {restaurant['address']}")
    if "phone" in user_input:
        print(f"Phone: {restaurant['phone']}")
    if "post" in user_input:
        print(f"Postal code: {restaurant['postcode']}")
    if "adress" in user_input:
        print(f"Address: {restaurant['addr']}")

# Find the restaurants that match the user preferences
def find_restaurants(preferences):
    matched_restaurants = []
    for _, row in db.iterrows():
        matched = 0
        for key, value in preferences.items():
            if value and row[key] == value:
                matched += 1
        if matched > 0 and row["restaurantname"] not in storage["exclude"]:
            matched_restaurants.append({"restaurant": row, "matched": matched})

    sorted_restaurants = sorted(
        matched_restaurants, key=lambda x: x["matched"], reverse=True
    )

    return [x["restaurant"] for x in sorted_restaurants]

# Helper function used to suggest a restaurant
def suggest_restaurant(current_state: DialogState, user_input: str, model_prediction: DialogAct):
    # find restaurants that match the user preferences
    restaurants = find_restaurants(storage["preferences"])

    # if no restaurants are found, ask for more preferences
    if not restaurants:
        reply("no_matches")
        return DialogState.ASK_PREFERENCES
    else:
        # store the current suggestion
        storage["current_suggestion"] = restaurants[0]

        # return the suggestion to the user
        reply("suggest_restaurant", restaurant=restaurants[0]["restaurantname"])
        return DialogState.OTHER_REQUEST

# Helper function used to ask for user preferences
def ask_preferences(current_state: DialogState, user_input: str, model_prediction: DialogAct):
    global storage

    # extract the preferences from the user input
    preferences, suggestions = extract_preferences(user_input)

    # update the storage with the new preferences
    update_storage("preferences", preferences)
    update_storage("suggestions", suggestions)

    if DEBUG:
        print(f"Preferences: {storage['preferences']}")
        print(f"Suggestions: {storage['suggestions']}")

    # check if there are any suggestions
    if any(suggestions.values()):
        for key, value in suggestions.items():
            if value:
                # ask the user if the suggestion is correct
                reply("ask_preferences", key=key, value=value)
                return DialogState.ASK_CLEAR_PREFERENCES

    # if we have enough preferences, suggest a restaurant
    if len(storage["preferences"]) >= 2:
        return suggest_restaurant(current_state, user_input, model_prediction)
    else:
        # ask the user for more preferences
        reply("ask_more_preferences")
        return DialogState.ASK_PREFERENCES

# The main dialog manager
# This function is called for each turn of the conversation
def dialog_manager(current_state: DialogState, user_input: str, model_prediction: DialogAct):
    global storage

    if DEBUG:
        print(f"Current state: {current_state}")
        print(f"User input: {user_input}")
        print(f"Model prediction: {model_prediction}")
        print(f"Current preferences: {storage['preferences']}")

    # handle the special dialog acts
    if model_prediction == DialogAct.NULL:
        # unclear input, ask the user to repeat
        reply("null")
        return current_state
    elif model_prediction == DialogAct.RESTART:
        # clear the storage and start over
        storage = {"preferences": {}, "suggestions": {}, "exclude": []}

        reply("welcome")
        return DialogState.WELCOME

    match current_state:
        case DialogState.WELCOME:
            if model_prediction == DialogAct.HELLO:
                # greet the user
                reply("welcome")
                return DialogState.WELCOME
            else:
                # ask for user preferences
                return ask_preferences(current_state, user_input, model_prediction)

        case DialogState.ASK_PREFERENCES:
            if model_prediction == DialogAct.INFORM:
                return ask_preferences(current_state, user_input, model_prediction)
            else:
                # the user did not provide any preferences, ask again
                reply("clear_preferences")
                return DialogState.ASK_PREFERENCES

        case DialogState.ASK_CLEAR_PREFERENCES:
            if model_prediction == DialogAct.ACK or model_prediction == DialogAct.AFFIRM:
                # update the preferences with the suggestions
                for key, value in storage["suggestions"].items():
                    storage["preferences"][key] = value
                    storage["suggestions"][key] = None

                # if we have enough preferences, suggest a restaurant
                if len(storage["preferences"]) >= 2:
                    return suggest_restaurant(current_state, user_input, model_prediction)

            # if the user does not agree with the suggestions, or gives new preferences, parse them
            return ask_preferences(current_state, user_input, model_prediction)

        case DialogState.OTHER_REQUEST:
            if model_prediction == DialogAct.NEGATE or model_prediction == DialogAct.REQALTS:
                # The user did not like the suggestion, exclude it and ask for more preferences
                storage["exclude"].append(storage["current_suggestion"]["restaurantname"])

                return ask_preferences(current_state, user_input, model_prediction)

            elif model_prediction == DialogAct.REQMORE:
                # The user asked for more restaurants, suggest the next best one
                storage["exclude"].append(storage["current_suggestion"]["restaurantname"])
                restaurants = find_restaurants(storage["preferences"])

                if restaurants:
                    storage["exclude"].append(restaurants[0]["restaurantname"])
                    reply("remove_option")
                    reply("suggest_restaurant", restaurant=restaurants[0]["restaurantname"])

                    return DialogState.OTHER_REQUEST
                else:
                    reply("no_more_matches")
                    return DialogState.ASK_PREFERENCES


            # the user asked for more information about the current suggestion
            elif model_prediction == DialogAct.REQUEST:
                get_more_info(storage["current_suggestion"], user_input)
                return DialogState.OTHER_REQUEST

            # the user is finished, say goodbye
            elif model_prediction == DialogAct.THANKYOU or model_prediction == DialogAct.BYE:
                reply("goodbye")
                return DialogState.GOODBYE

            else:
                return DialogState.OTHER_REQUEST

        # the conversation is finished
        case DialogState.GOODBYE:
            if model_prediction == DialogAct.BYE:
                reply("goodbye")
                return None
            else:
                return DialogState.GOODBYE


def main():
    # initialize the dialog manager
    current_state = DialogState.WELCOME
    reply("greeting")

    while current_state:
        # get user input
        user_input = input().lower()

        # get the model prediction
        model_input = count_vectorizer.transform([user_input])
        prediction = model.predict(model_input)
        action = DialogAct(prediction)


        if DEBUG:
            print(f"---- Model prediction: {prediction} ({action})")

        # call the dialog manager
        current_state = dialog_manager(current_state, user_input, action)

        if DEBUG:
            print(f"---- Output_state: {current_state}")


if __name__ == "__main__":
    main()
