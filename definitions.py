d1={
        "water":"",
        "hey how are u":"I am well how are u",
        "how are u":"I am great  how can i assist u",
        "hey hello": "Here to assist u",
        "Hi chatbot": "Hey user how can i assist u",
        "listen chatbot": "Yes  How may i help you ",
        "chatbot listen":"yes i am listening",
        "chatbot pls listen":"yes i want to listen",
        "i need your help":"how may i assist you",
        "i want you to help me":"how can i help you",
        "i need you to assist me": "how may i assist you",
        "help me chatbot":"yes sir how may i assist you",
        "chatbot how are you":"fine sir what about you",
        "my product is damaged":"ok sir provide me detail of your product",
        "product damaged":"want to replace damaged product or not ",
        "product recieved":"hope you like the product",
        "product replacemnt":"sure provide more details about product",
        "product review":"sure here is the link to review you code",
        "unsatisfied with product":"please provide me more details",
        "late delivery":"so should i deliver it to you",
        "i want early delivery":"would you like to go with fedx or fedplane"

    }


def create_dictionary(key_file, value_file):
        # Initialize an empty dictionary
        my_dictionary = {}

        # Open both files
        with open(key_file, 'r') as keys, open(value_file, 'r') as values:
                # Read lines simultaneously from both files
                for key_line, value_line in zip(keys, values):
                        # Strip newline characters and whitespace
                        key_line = key_line.strip()
                        value_line = value_line.strip()

                        # Add key-value pair to dictionary
                        my_dictionary[key_line] = value_line

        return my_dictionary