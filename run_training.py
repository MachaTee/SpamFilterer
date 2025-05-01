import doublechain as m
# import model as y

model = m.train() 
# model = m.train() 

# ymodel = y.train()
# ymodel = y.train()

# for x in range(60, 101):
    # discriminant = m.discriminate(spam_threshold = x/100)

model = m.calculate_probability(model)
discriminant = m.discriminate()
# ydiscriminant = y.discriminate()