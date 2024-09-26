import pandas as pd
from enum import Enum
from Levenshtein import distance

class DialogState(Enum):
    WELCOME = 1  # starting state 
    ASK_PREFERENCES = 2  # User provides preferences
    ASK_CLEAR_PREFERENCES = 3  # Ask for more clear answers
    NO_RESTAURANT_FOUND = 4  # 404 No restaurant found
    SUGGEST_RESTAURANT = 5  # Suggest a restaurant somehow
    DELETE_DISLIKED_RESTAURANT = 6  # User dislikes the suggestion. delete it
    OTHER_REQUEST = 7  #       other user request
    GOODBYE = 8  # End 

db = pd.read_csv('restaurant_info.csv')

options = {
    "food": db["food"].dropna().unique(),
    "pricerange": db["pricerange"].dropna().unique(),
    "area": db["area"].dropna().unique()
}

keywords = {
    "food": ["food", "cuisine"],
    "pricerange": ["price", "cost"],
    "area": ["location", "area"]
}

def extract_preferences(user_input):
    splitted_input = user_input.split()
    preferences = {"food": None, "pricerange": None, "area": None}
    suggestions = {"food": None, "pricerange": None, "area": None}

    for key, values in keywords.items():
        min_distance = 1000
        word = None

        for i, value in enumerate(splitted_input):
            if value in values:
                word = splitted_input[i - 1]
                break

        if word:
            if word in options[key]:
                preferences[key] = word
            else:
                for option in options[key]:
                    if distance(word, option) < min_distance:
                        min_distance = distance(word, option)
                        suggestions[key] = option


    return (preferences, suggestions)


def find_restaurants(preferences):
    matched_restaurants = []
    for _, row in db.iterrows():
        matched = 0
        for key, value in preferences.items():
            if value and row[key] == value:
                matched += 1
        if matched > 0:
            matched_restaurants.append({ "restaurant": row, "matched": matched })

    sorted_restaurants = sorted(matched_restaurants, key=lambda x: x["matched"], reverse=True)
    return [x["restaurant"] for x in sorted_restaurants]

def dialog_manager(current_state, user_input):
    preferences = {"food_type": None, "pricerange": None, "area": None}
    if current_state == DialogState.WELCOME:
        print("Welcome! What type of restaurant are you looking for?")
        return DialogState.ASK_PREFERENCES
    
    elif current_state == DialogState.ASK_PREFERENCES:
        (preferences, suggestions) = extract_preferences(user_input)
        # check was there any preferences
        if suggestions["food"] or suggestions["pricerange"] or suggestions["area"]:
            for key, value in suggestions.items():
                if value:
                    print(f"I'm sorry, I didn't understand. Did you mean {value} for {key}?")
                    user_input = input().lower()
                    if user_input == "no":
                        return DialogState.ASK_CLEAR_PREFERENCES
                    else:
                        preferences[key] = value

        if not ( preferences["food"] or preferences["pricerange"] or preferences["area"] ):
            print("I'm sorry, I didn't understand. Could you please provide a more clear answer?")
            return DialogState.ASK_CLEAR_PREFERENCES
        else:  
            matched_restaurants = find_restaurants(preferences)
            if matched_restaurants:
                print(f"Here is a recommendation: {matched_restaurants[0]['restaurantname']}.")
                return DialogState.SUGGEST_RESTAURANT
            else:
                print("Sorry, no restaurant matches your preferences.")
                return DialogState.NO_RESTAURANT_FOUND
        
    
    elif current_state == DialogState.ASK_CLEAR_PREFERENCES:
        preferences = extract_preferences(user_input)
        if preferences["food"] or preferences["pricerange"] or preferences["area"]:
            return DialogState.ASK_PREFERENCES
        else:
            print("I still couldn't understand. Could you please try again?")
            return DialogState.ASK_CLEAR_PREFERENCES

    elif current_state == DialogState.SUGGEST_RESTAURANT:
        print("Would you like this restaurant? If not, I can suggest another.")
        return DialogState.DELETE_DISLIKED_RESTAURANT
    
    elif current_state == DialogState.DELETE_DISLIKED_RESTAURANT:
        if user_input == "no":
            print("Removing that option, let me suggest another.")
            # Logic to suggest another restaurant
            return DialogState.SUGGEST_RESTAURANT
        else:
            print("Great! I'm glad you like it.")
            return DialogState.OTHER_REQUEST
    
    elif current_state == DialogState.NO_RESTAURANT_FOUND:
        print("Unfortunately, no restaurants matched your preferences.")
        preferences = {"food": None, "pricerange": None, "area": None}
        return DialogState.ASK_PREFERENCES
    
    elif current_state == DialogState.OTHER_REQUEST:
        print("Do you have any other requests or preferences?")
        # Handle other requests or end the dialog
        return DialogState.GOODBYE
    
    elif current_state == DialogState.GOODBYE:
        print("Goodbye! Have a nice day.")
        return None  # End the conversation


def main():
    current_state = DialogState.WELCOME
    while current_state:
        user_input = input().lower()
        current_state = dialog_manager(current_state, user_input)

if __name__ == "__main__":
    main()
