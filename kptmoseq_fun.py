def contains(substring, string_list):
    # Iterate over each element in the list
    for item in string_list:
        # Check if the substring is part of the current item
        if substring in item:
            return True  # Return True if the substring is found
    return False  # Return False if the substring is not found in any list item
