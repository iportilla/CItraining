test_url_mal = "naureen.net/etisalat.ae/index2.php"
test_url_benign = "sixt.com/php/reservation?language=en_US"
url = test_url_benign
#url = test_url_mal
# Step 1: Convert raw URL string in list of lists where characters that are contained in "printable" are stored encoded as integer
url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
max_len=75
T = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
target_proba = model.predict(T, batch_size=1)
def print_result(proba):
    if proba > 0.5:
        return "malicious"
    else:
        return "benign"
print("Test URL:", url, "is", print_result(target_proba[0]))
