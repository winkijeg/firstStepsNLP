
from nltk.corpus import wordnet as wn


def main():
    my_synset = wn.synsets('dog')
    print my_synset

    for ss in wn.synsets('dog'):
        print ss, ss.definition()



if __name__ == '__main__':
    main()
