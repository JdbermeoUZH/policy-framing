import difflib
from termcolor import colored


# Let's define a function that lets us examine the effects of processing
# We will pass the text to process, and the processing method we want to use
def before_and_after(text, method, kwargs={}):

  # get the before and after, then compare them with difflib's SequenceMatcher
  before = text
  after = method(text,**kwargs)
  if type(after) == list:
    after = ' '.join(after)
  diffs = difflib.SequenceMatcher(None, before, after)

  # This part takes a string and some sections that do match the other string
  # but highlights the parts that *don't* match
  def print_highlight_nonmatching(text, text_id, matches, hl_func):
    current = 0
    for match in matches:
      nonmatching_count = match[text_id] - current
      # some part of the string didn't match the other,
      # need to print it with hl_func
      if nonmatching_count:
        substring_end = current + nonmatching_count
        print(hl_func(text[current:substring_end]),end='')
        current = substring_end
      # now print the part that *did* match without any color effects
      substring_end = current + match.size
      print(text[current:substring_end],end='')
      current = substring_end
    # print a blank line between the two
    print()

  # this will get the sequences that did match between the two strings
  matches = diffs.get_matching_blocks()
  hl_func_red = lambda x: colored(x,'red')
  hl_func_blue = lambda x: colored(x,'blue')

  # now use this to show the before and after with changes highlighted
  print("BEFORE: ",end="")
  print_highlight_nonmatching(before, 0, matches, hl_func_red)
  print("AFTER:  ",end="")
  print_highlight_nonmatching(after, 1, matches, hl_func_blue)