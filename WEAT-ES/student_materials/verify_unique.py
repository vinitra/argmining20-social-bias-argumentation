# avoid repeat words after translation
with open('./wordlists/negative-words.txt', 'r') as f:
    seen = {}
    for line in f:
        if line != line.lower():
            print(line.strip())
        if line in seen:
            print(line.strip())
        else:
            seen[line] = True
