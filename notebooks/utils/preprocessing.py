import html
import re


def remove_whitespace(content):
    content = re.sub(r'( |\xa0)+', ' ', content)
    return '\n'.join([s for s in content.splitlines() if s.strip()])


def remove_pattern(content, regex, replace_with=''):
    pattern = re.compile(regex)
    while True:
        m = re.search(pattern, content)
        if m is None:
            break
        content = content[:m.start(0)] + replace_with + content[m.end(0):]
    return content


def replace_html_special_ents(content):
    pattern = re.compile(r'&#\d{1,4};|&\w{1,6};')
    while True:
        m = re.search(pattern, content)
        if m is None:
            break
        unicode = html.unescape(m.group(0))
        content = content[:m.start(0)] + unicode + content[m.end(0):]
    return content
