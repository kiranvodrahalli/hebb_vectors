#!/usr/bin/env python
# -*- coding: utf-8 -*-
import enchant

def build_french():
	delstr_f = ".,;:?!()[]\\/"
	f = open("hp1_fr.txt", "rb")
	fr_str = ""
	for line in f:
		fr_str += line + " "
	f.close()
	fr_str0 = fr_str.translate(None, delstr_f)
	fr_str1 = fr_str0.replace("È", "è")
	fr_str2 = fr_str1.replace("«", "")
	fr_str3 = fr_str2.replace("»", "")
	fr_str4 = fr_str3.replace("\n", " ")
	fr_str5 = fr_str4.replace("\t", " ")
	fr_str6 = fr_str5.replace("-t-", " ")
	fr_str7 = fr_str6.replace("-", " ")
	fr_str8 = fr_str7.replace("-t-", " ")
	fr_str9 = fr_str8.replace("—", " ")
	fr_str10 = fr_str9.replace("œ", "oe")
	fr_str11 = fr_str10.replace("É", "é")
	fr_str12 = fr_str11.replace("qu'", "que ")
	fr_str13 = fr_str12.replace("d'", "de ")
	fr_str14 = fr_str13.replace("l'", "le ")
	fr_str15 = fr_str14.replace("j'", "je ")
	fr_str16 = fr_str15.replace("m'", "me ")
	fr_str17 = fr_str16.replace("n'", "ne ")
	fr_str18 = fr_str17.replace("s'", "se ")
	fr_str19 = fr_str18.replace("Ï", "ï")
	fr_words = fr_str19.split(" ")
	fr_words0 = []
	for w in fr_words:
		if w != '':
			fr_words0.append(w.lower())
	fr_words1 = fix_fr(fr_words0)
	return fr_words1
# fix to make sensible words/ split up combined words
def fix_fr(fr_words0):
	fr_words1 = []
	for w in fr_words0:
		if w == "ro":
			fr_words1.append("rogue")
		elif w == "'mande":
			fr_words1.append("demande")
		elif w == "attendezmoi":
			fr_words1.append("attendez")
			fr_words1.append("moi")
		elif w == "dumbled":
			fr_words1.append("dumbledore")
		elif w == "tusais":
			fr_words1.append("tusaisqui")
		elif w == "til":
			fr_words1.append("il") # throw out the t
		elif w == "saisqui":
			fr_words1.append("tusaisqui")
		elif w == "s'rez":
			fr_words1.append("serez")
		elif w == "s'fier":
			fr_words1.append("se")
			fr_words1.append("fier")
		elif w == 'queloue':
			fr_words1.append("quelque")
		elif w == "n'suis":
			fr_words1.append("ne")
			fr_words1.append("suis")
		elif w == "maisnonmaisquoimaispasdutout":
			fr_words1.append("mais")
			fr_words1.append("non")
			fr_words1.append("mais")
			fr_words1.append("quoi")
			fr_words1.append("mais")
			fr_words1.append("pas")
			fr_words1.append("du")
			fr_words1.append("tout")
		elif w == "làdedans":
			fr_words1.append("là")
			fr_words1.append("dedans")
		elif w == "làhaut":
			fr_words1.append("là")
			fr_words1.append("haut")
		elif w == "ii":
			fr_words1.append("il")
		elif w == "i'immobiliser":
			fr_words1.append("le")
			fr_words1.append("immobiliser")
		elif w == "genslà":
			fr_words1.append("gens")
			fr_words1.append("là")
		elif w == 'estce':
			fr_words1.append("est")
			fr_words1.append("ce")
		elif w == 'etait':
			fr_words1.append("était")
		elif w == 'etrange':
			fr_words1.append("étrange")
		elif w == 'etude':
			fr_words1.append("étude")
		elif w == 'etre':
			fr_words1.append("être")
		elif w == 'ditil':
			fr_words1.append("dit")
			fr_words1.append("il")
		elif w == "concen":
			fr_words1.append("concentrer")
		elif w == "c'que":
			fr_words1.append("ce")
			fr_words1.append("que")
		elif w == "ca":
			fr_words1.append("ça")
		elif w == "aujourde":
			fr_words1.append("aujourd'hui")
		elif w == "hui":
			print w
		elif w == "échapp'rien":
			fr_words1.append("échappé")
			fr_words1.append("rien")
		elif w == "l'":
			fr_words1.append("le")
		elif w == "l'elevage":
			fr_words1.append("le")
			fr_words1.append("elevage") # split it up
		elif w == "nedoit":
			fr_words1.append("ne")
			fr_words1.append("doit")
		elif w == "v" or w == "r" or w == "p" or w == "m" or w == "j" or w == "h" or w == "f" or w == "g" or w == "c" or w == "b":
			print w
		else: fr_words1.append(w)
	return fr_words1
def check_nonwords(fr_words0):
	d_fr = enchant.DictWithPWL("fr", "hp_words_fr.txt")
	# still will need to add some words to a text file that are hp-specific
	non_words = set()
	for w in fr_words0:
		if not d_fr.check(w):
			non_words.add(w)
	print len(non_words)
	return non_words

def find_context(to_find, fr_words0):
	for i in range(0, len(fr_words0)):
		w = fr_words0[i]
		if w == to_find:
			for j in range(i - 10, i +11):
				print fr_words0[j]
			print "----------------"

# use to fix the typos from OCR when converting from pdf to txt
def fix(hp1_words2):
	hp1_words3 = []
	for w in hp1_words2:
		if w == "1ude":
			hp1_words3.append("rude")
		elif w == 'Giunnings':
			hp1_words3.append("Grunnings")
		elif w == 'Gryfindor':
			hp1_words3.append("Gryffindor")
		elif w == 'Gryfizndors':
			hp1_words3.append("Gryffindors")
		elif w == 'Hufiflepufif':
			hp1_words3.append("Hufflepuff")
		elif w == 'Hufilepufls':
			hp1_words3.append("Hufflepuffs")
		elif w == "Im":
			hp1_words3.append("I")
			hp1_words3.append("am")
		elif w == "Ipromised":
			hp1_words3.append("I")
			hp1_words3.append("promised")
		elif w == "Iwill":
			hp1_words3.append("I")
			hp1_words3.append("will")
		elif w == 'L\xc2\xa2Why99':
			hp1_words3.append("Why")
		elif w == 'OTALLOWED':
			hp1_words3.append("NOT")
			hp1_words3.append("ALLOWED")
		elif w == 'Onlyjoking':
			hp1_words3.append("Only")
			hp1_words3.append("joking")
		elif w == 'Petrmcus':
			hp1_words3.append("Petrificus")
		elif w == 'Pleasefind':
			hp1_words3.append("Please")
			hp1_words3.append("find")
		elif w == 'Quidditchfield':
			hp1_words3.append("Quidditch")
			hp1_words3.append("field")
		elif w == 'Thanksfor':
			hp1_words3.append("Thanks")
			hp1_words3.append("for")
		elif w == 'WITCHCRAF':
			hp1_words3.append("WITCHCRAFT")
		elif w == 'Wafiling':
			hp1_words3.append("Waffling")
		elif w == 'Wargr':
			hp1_words3.append("Warty")
		elif w == 'Yes79':
			hp1_words3.append("Yes")
		elif w == 'Yourejoking':
			hp1_words3.append("Youre")
			hp1_words3.append("joking")
		elif w == 'Yourfather':
			hp1_words3.append("Your")
			hp1_words3.append("father")
		elif w == 'abou':
			hp1_words3.append("about")
		elif w == 'acreeping':
			hp1_words3.append("creeping")
		elif w == 'aflapl':
			hp1_words3.append("a")
			hp1_words3.append("flap")
		elif w == 'allpupils':
			hp1_words3.append("all")
			hp1_words3.append("pupils")
		elif w == 'alwaysfind':
			hp1_words3.append("always")
			hp1_words3.append("find")
		elif w == 'andfiftyeight':
			hp1_words3.append("and")
			hp1_words3.append("fiftyeight")
		elif w == 'andfull':
			hp1_words3.append("and")
			hp1_words3.append("full")
		elif w == 'anywa':
			hp1_words3.append("anyway")
		elif w == 'aren':
			hp1_words3.append("arent")
		elif w == 'atfirst':
			hp1_words3.append("at")
			hp1_words3.append("first")
		elif w == 'birdsflying':
			hp1_words3.append("birds")
			hp1_words3.append("flying")
		elif w == 'blackfor':
			hp1_words3.append("black")
			hp1_words3.append("for")
		elif w == 'c1utches':
			hp1_words3.append("crutches")
		elif w == 'carefiilly':
			hp1_words3.append("carefully")
		elif w == 'carefirl':
			hp1_words3.append("careful")
		elif w == 'clockfor':
			hp1_words3.append("clock")
			hp1_words3.append("for")
		elif w == 'closern':
			hp1_words3.append("closer")
		elif w == 'cluesfour':
			hp1_words3.append("clues")
			hp1_words3.append("four")
		elif w == 'confiised':
			hp1_words3.append("confused")
		elif w == 'crystalphials':
			hp1_words3.append("crystal")
			hp1_words3.append("phials")
		elif w == 'differentcolored':
			hp1_words3.append("different")
			hp1_words3.append("colored")
		elif w == 'differentsized':
			hp1_words3.append("different")
			hp1_words3.append("sized")
		elif w == 'diffrent':
			hp1_words3.append("different")
		elif w == 'dontjudge':
			hp1_words3.append("dont")
			hp1_words3.append("judge")
		elif w == 'dqferent':
			hp1_words3.append("different")
		elif w == 'safegr':
			hp1_words3.append("safety")
		elif w == 'dwarfnor':
			hp1_words3.append("dwarf")
			hp1_words3.append("nor")
		elif w == 'endofyear':
			hp1_words3.append("end")
			hp1_words3.append("of")
			hp1_words3.append("year")
		elif w == 'firll':
			hp1_words3.append("full")
		elif w == 'firnny':
			hp1_words3.append("funny")
		elif w == 'g0':
			hp1_words3.append("go")
		elif w == 'gratefirl':
			hp1_words3.append("grateful")
		elif w == 'halflreartedly':
			hp1_words3.append("halfheartedly")
		elif w == 'himselfa':
			hp1_words3.append("himself")
		elif w == 'lattened':
			hp1_words3.append("flattened")
		elif w == 'lqe':
			hp1_words3.append("life")
		elif w == 'lufiy':
			hp1_words3.append("Fluffy")
		elif w == 'manciushing':
			hp1_words3.append("mancrushing")
		elif w == 'myselfifyou':
			hp1_words3.append("myself")
			hp1_words3.append("if")
			hp1_words3.append("you")
		elif w == 'mystry':
			hp1_words3.append("mystery")
		elif w == 'nvelve':
			hp1_words3.append("twelve")
		elif w == 'ofGreat':
			hp1_words3.append("of")
			hp1_words3.append("Great")
		elif w == 'ofLqe':
			hp1_words3.append("of")
			hp1_words3.append("Life")
		elif w == 'ofMagic':
			hp1_words3.append("of")
			hp1_words3.append("Magic")
		elif w == 'ofMerlin':
			hp1_words3.append("of")
			hp1_words3.append("Merlin")
		elif w == 'ofSpells':
			hp1_words3.append("of")
			hp1_words3.append("Spells")
		elif w == 'ofair':
			hp1_words3.append("of")
			hp1_words3.append("air")
		elif w == 'ofall':
			hp1_words3.append("of")
			hp1_words3.append("all")
		elif w == 'ofeach':
			hp1_words3.append("of")
			hp1_words3.append("each")
		elif w == 'offlujf':
			hp1_words3.append("of")
			hp1_words3.append("fluff")
		elif w == 'ofglass':
			hp1_words3.append("of")
			hp1_words3.append("glass")
		elif w == 'ofowls':
			hp1_words3.append("of")
			hp1_words3.append("owls")
		elif w == 'ofmine':
			hp1_words3.append("of")
			hp1_words3.append("mine")
		elif w == 'ofplain':
			hp1_words3.append("of")
			hp1_words3.append("plain")
		elif w == 'ofprotective':
			hp1_words3.append("of")
			hp1_words3.append("protective")
		elif w == 'ofsightings':
			hp1_words3.append("of")
			hp1_words3.append("sightings")
		elif w == 'oftea':
			hp1_words3.append("of")
			hp1_words3.append("tea")
		elif w == 'ofthe':
			hp1_words3.append("of")
			hp1_words3.append("the")
		elif w == 'ofthefollowing':
			hp1_words3.append("of")
			hp1_words3.append("the")
			hp1_words3.append("following")
		elif w == 'ofthese':
			hp1_words3.append("of")
			hp1_words3.append("these")
		elif w == 'oftoil':
			hp1_words3.append("of")
			hp1_words3.append("toil")
		elif w == 'ofus':
			hp1_words3.append("of")
			hp1_words3.append("us")
		elif w ==  'ourfloors':
			hp1_words3.append("our")
			hp1_words3.append("floors")
		elif w == 'ojf':
			hp1_words3.append("off")
		elif w == 'ofif':
			hp1_words3.append("off")
		elif w == 'painfiil':
			hp1_words3.append("painful")
		elif w == "ordan":
			hp1_words3.append("Jordan")
		elif w == 'offlrand':
			hp1_words3.append("offhand")
		elif w == 'qyou':
			hp1_words3.append("you")
		elif w == 'qyoure':
			hp1_words3.append("youre")
		elif w == 'qyouve':
			hp1_words3.append("youve")
		elif w == 'realfriends':
			hp1_words3.append("real")
			hp1_words3.append("friends")
		elif w == 'refirsing':
			hp1_words3.append("refusing")
		elif w == 'sawand':
			hp1_words3.append("wand")
		elif w == 'sc1uffs':
			hp1_words3.append("scruffs")
		elif w == 'scaredlooking':
			hp1_words3.append("scared")
			hp1_words3.append("looking")
		elif w == 'schoolfor':
			hp1_words3.append("school")
			hp1_words3.append("for")
		elif w == 'sha1p':
			hp1_words3.append("sharp")
		elif w == 'sha1ply':
			hp1_words3.append("sharply")
		elif w == 'silverfastenings':
			hp1_words3.append("silver")
			hp1_words3.append("fastenings")
		elif w == 'somefriends':
			hp1_words3.append("some")
			hp1_words3.append("friends")
		elif w == 'stufif':
			hp1_words3.append("stuff")
		elif w == 't1ust':
			hp1_words3.append("trust")
		elif w == 'talkingto':
			hp1_words3.append("talking")
			hp1_words3.append("to")
		elif w == 'thankyou':
			hp1_words3.append("thank")
			hp1_words3.append("you")
		elif w == 'thirdfloor':
			hp1_words3.append("third")
			hp1_words3.append("floor")
		elif w == 'truthfirl':
			hp1_words3.append("truthful")
		elif w == 'undred':
			hp1_words3.append("hundred")
		elif w == 've':
			hp1_words3.append("have")
		elif w == 'wor':
			hp1_words3.append("work")
		elif w == 'Iuins':
			hp1_words3.append("ruins")
		elif w == 'Iumbling':
			hp1_words3.append("rumbling")
		elif w == 'Iuffled':
			hp1_words3.append("ruffled")
		elif w == 'moren':
			hp1_words3.append("more")
			hp1_words3.append("than")
		elif w == 'aff':
			hp1_words3.append("afford")
		elif w == 'Iby':
			hp1_words3.append("One")
			hp1_words3.append("by")
		elif w == 'un':
			hp1_words3.append("one")
		elif w == 'sc':
			hp1_words3.append("score")
		elif w == 'concen':
			hp1_words3.append("concentrate")
		elif w == "B" or w == "C" or w == "D" or w == "E" or w == "F" or w == "G" or w == "H" or w == "J" or w == "K" or w == "L" or w == "M" or w == "N" or w == "O" or w == "P" or w == "Q" or w == "R" or w == "S" or w == "T" or w == "U" or w == "V" or w == "X" or w == "Y" or w == "Z":
			print w
		else:
			hp1_words3.append(w)
	return hp1_words3

