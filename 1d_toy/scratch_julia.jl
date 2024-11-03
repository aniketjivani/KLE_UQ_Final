using PyCall

# # Import necessary Python modules
# importlib = pyimport("importlib")
sys = pyimport("sys")

# # Set the command-line arguments as desired
sys.argv = ["./1d_toy/scratch_python.py",
            "Hello", 
            "World"]

# # Import the script as a module
# mod = importlib.import_module("scratch_python")

# # Call the main function
# mod.main("Hello", "World")

# pyinclude(fname) = (PyCall.pyeval_(read(fname, String), PyCall.pynamespace(Main), PyCall.pynamespace(Main), PyCall.Py_file_input, fname); nothing)


# pyinclude("1d_toy/scratch_python.py")

@pyinclude "1d_toy/scratch_python.py"

result = py"result"
computed_value = py"computed_value"