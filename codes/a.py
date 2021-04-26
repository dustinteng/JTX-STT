folder = '/videos/test0'
for count in range(6):
    string = folder +  "/%#05d.jpg" % (count+1)
    print string