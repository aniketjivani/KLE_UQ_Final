# # my_script.py
# def main(arg1, arg2):
#     result = f"{arg1} and {arg2} from Python"
#     computed_value = len(arg1) + len(arg2)  # Just an example computation
#     # return result, computed_value  # Return multiple values as a tuple
#     return computed_value

# if __name__ == "__main__":
#     import sys
#     main(sys.argv[1], sys.argv[2])

import sys

result = f"{sys.argv[1]} and {sys.argv[2]} from Python"
computed_value = len(sys.argv[1]) + len(sys.argv[2])  # Just an example computation
