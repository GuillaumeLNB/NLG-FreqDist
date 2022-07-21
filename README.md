dummy text generator using word frequencies

under construction

```
>>> from nlg_freq_dist import NgramGenerator

>>> with open("./data/hp-en.txt", "r") as f:
        text = f.read()
>>> model = NgramGenerator(text)
>>> # model.generate_text("Harry wanted to see", 2)
>>> print(model.generate_text_deterministic("Harry wanted to see", 3))
Harry wanted to see if anyone found out about the Stone , it 's a unicorn , we 'll be in Slytherin , '' said Harry . `` I 'm not going to be a bit of toast before going off to the door , and the next morning . He was n't going to be a bit of toast before going off to the door , and the next morning . He was n't going to be a bit of toast before going off to the door , and the next morning . He was n't going to be a bit of toast
>>> # print(model.generate_text("Harry wanted to see", 3))
>>> # model.generate_text("Harry wanted to see", 4)
```