# from nlg_freq_dist import NgramGenerator

# with open("../data/hp-en.txt") as f:
#     text = f.read()

# model = NgramGenerator(text)
# # creating 10 tokens long text based on tri-grams
# model.generate_text("Harry wanted", 2, 10)
# # creating 40 tokens long text based on tri-grams
# model.generate_text("Harry wanted", 3, 40)
# # creating 100 tokens long text based on 4-grams
# model.generate_text("Harry wanted", 4, 100)

from nlg_freq_dist import NgramGenerator

with open("./data/hp-en.txt", "r") as f:
    text = f.read()
model = NgramGenerator(text)
# model.generate_text("Harry wanted to see", 2)
print(model.generate_text_deterministic("Harry wanted to see", 3))
# print(model.generate_text("Harry wanted to see", 3))
# model.generate_text("Harry wanted to see", 4)
