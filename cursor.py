#Additional utility function for MacOS to show and hide the terminal cursor 

def hide():
    """Hide the terminal cursor."""
    print("\033[?25l", end="", flush=True)

def show():
    """Show the terminal cursor."""
    print("\033[?25h", end="", flush=True)
