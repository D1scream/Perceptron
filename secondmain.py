def find_expensive_items(prices_string):
    prices = [float(x) for x in prices_string.split(",")]
    av = sum(prices) / len(prices)
    l=""
    for i in prices:
        if i>av:
            l+=f"{i}".replace(".0","")
            l+="\n"
    return l
    

prices_string = "5, 5, 5, 5"
result = find_expensive_items(prices_string)
print(result)
