
istream = open('Le_petit_prince.u8.lemma')

bfr = []
for line in istream:
    
    toklist = [tok.split('/')[0].replace('_',' ') for tok in line.split()]

    #sentence splitting:
    for t in toklist:
        if t in ['...','!','?','.']:
            bfr.append(t)
            print(' '.join(bfr))
            bfr = []
        else:
            bfr.append(t)
    
if bfr:
    print(' '.join(bfr))

istream.close()
