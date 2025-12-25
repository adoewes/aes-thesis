import language_tool_python
tool = language_tool_python.LanguageTool('en-US')  # use a local server (automatically set up), language English

def get_language_errors(answer):
    matches = tool.check(answer)
    #print("no. of language errors: ", len(matches))
    return len(matches)

