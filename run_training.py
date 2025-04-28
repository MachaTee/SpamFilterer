import model as m 
model = m.train() 

for x in range(50, 100):
    discriminant = m.discriminate(spam_threshold = x/100)