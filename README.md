
# Active Learning with Python Tutorial

This tutorial is for using Active Learning in Python for a text dataset. The code is based off of Byron Wallace's (UT Austin) Python module, with my ammendments for Pandas functionality, of libsvm
with the SIMPLE querying function of Tong & Koller (2001). The code
was adjusted to for querying where the user is asked
to classify nonlabeled points which best halve the version space. 

1. Download Byron's code, https://github.com/pesoto/curious_snake

2. Extract files to some folder on your local drive

3. CD into .../libsvm/python/ and type 'make' to build the module (or 'python setup.py install')

4. Download my additions, along with the movie review data for the demo below, https://github.com/pesoto/ActiveLearningAdditions

5. Export this folder into the folder you would like to work in

Note, this module requires Stephen Hansen's Topic Modelling module available at https://github.com/sekhansen/text-mining-tutorial


##Movie Review Demo

First, let's import the module and the data. 


    import learner
    import machine_learning
    import pandas as pd
    import numpy as np
    
    ### Import Data ###
    filename = 'review_data.csv'
    data = pd.read_csv(filename,encoding="utf-8")

The data is formatted with the original text, and the prelabeled classification. However, we will simulate a more realistic situation where only a few observations are really labeled. 


    print data.head()

                                                    text sentiment
    0  moviemaking is a lot like being the general ma...       pos
    1  jay and silent bob strike back , kevin smith's...       pos
    2  metro i've seen san francisco in movies many t...       pos
    3  when people are talking about good old times ,...       pos
    4  50's `aliens vs . \nearth' idea revamped ! \ni...       pos


###Pre-process Data

Now let's create the attributes. For this demo, we will use the TFIDF weights, but we could just as easily use the counts of each word in the vocabulary as the attributes with the TextDoc.bagofwords dataframe.


    ###################################################
    ##Using Bag of Words (Word Vectors) as attributes##
    ###################################################
    
    #Select 100 observations from the positive reviews
    #Select 100 observations from the negative reviews
    all_data = pd.concat([data.ix[0:100],data.ix[500:600]])
    #Initiate TextDoc object
    textData = machine_learning.TextDoc(all_data)
    #Generate Term Frequency-Inverse Document frequencies of Word Vectors (Note: this takes a bit of time)
    textData.tfidf()
    #Scale the features so each vector is of unit modulus
    textData.tfidf_df = textData.tfidf_df.apply(lambda x: x/np.linalg.norm(x),1)
    #Include dummy variables for each class label in the dataframe
    #####1 - Positive#####
    #####0 - Negative#####
    textData.tfidf_df["classLabel"] = pd.get_dummies(all_data['sentiment'])['pos']
    #Include the original text in the tfidf-dataframe
    textData.tfidf_df["origText"] = all_data.text

Let's start off with only two examples to start the querying- one from the class label 'positive', and one from the class label 'negative'. Then, we will use 50 observations for the test dataset and 150 observations for the active learner to choose from to join our training dataset. 


    #Choose 1 from each class- positive & negative
    labeled_data = textData.tfidf_df.loc[[0,500]]
    
    #Shuffle the remaining dataset
    shuffle = textData.tfidf_df.loc[np.random.permutation(textData.tfidf_df[~textData.tfidf_df.index.isin([0,500])].index)]
    
    #Use 150 for the pool of unlabeled, and 50 for the test data
    unlabeled_data = shuffle[0:150]
    test_data = shuffle[150::]

Now, we will initiate the ActiveLearningDataset object, which facilitates adding observations, dropping observations and undersampling minority examples. Note, we need to provide the variable name of the class variable as well as the original text.


    data1 = machine_learning.ActiveLearningDataset(labeled_data,classLabel="classLabel",origText="origText")
    data2 = machine_learning.ActiveLearningDataset(unlabeled_data,classLabel="classLabel",origText="origText")
    data3 = machine_learning.ActiveLearningDataset(test_data,classLabel="classLabel",origText="origText")

###Create Learner

Next, initiate the learner object which will execute the active learning and estimate the parameters of the support vector machine. 

The learner uses Support Vector Machines to find the most 'informative' unlabeled observations to train. The training, however, does not need to use SVMs. Instead, we can obtain our accuracy by using another classifier, such as the 
NOTE: In this demo, we are using the decision values to estimate the class for the out of sample data and are not estimating posterior probabilities for each class. If you'd like to use them, keep in mind the estimation takes much longer to build the models. Furthermore, accuracy will require more training data. 


    active_learner = learner.learner(data1,test_datasets=data3,probability=0,NBC=True)
    length = len(data1.data)
    active_learner.pick_initial_training_set(length)
    active_learner.rebuild_models(undersample_first=True)

    building models...
    training model(s) on 2 instances
    finding optimal C, gamma parameters...
    C:0.0625; gamma:0.0009765625
    positives found during learning: 0
    negatives found during learning: 0
    evaluating learner over 50 instances.
    confusion matrix:
    {'fp': 6, 'tn': 22, 'fn': 19, 'tp': 3}
    done.
    done.
    undersampling before building models..
    undersampling majority class to equal that of the minority examples
    removing 0 majority instances
    done.
    training model(s) on 2 instances
    finding optimal C, gamma parameters...
    C:0.0625; gamma:0.0009765625
    positives found during learning: 0
    negatives found during learning: 0
    evaluating learner over 50 instances.
    confusion matrix:
    {'fp': 6, 'tn': 22, 'fn': 19, 'tp': 3}
    done.


Right off the bat, the accuracy rate is 50% using the two observations as the training data, and the 50 observations in our test set. The confusion matrix reports the number of false positives ('fp'), true negatives ('tn'), false negatives ('fn') and true positives ('tp'). In other words, the true postivive/negatives are the ones the classifier correctly predicted, whereas the false positive/negative were the ones in which the classifier got it backwards. 

The 'C' parameter trades off error with stability. A high C increases the number of support vectors the classifier chooses and puts more weight on correctly guessing the training sample classes.

The 'gamma' parameter is that of the radial basis function which we set by default as the kernel function. 

The undersample_first = True evens the number of 'positive' and 'negative' reviews in estimation of the parameters to correct for biases in the original training dataset.

Next, we'll add the unlabeled data to the learner. The learner will query among these to ask the user to label the review it chooses.


    active_learner.unlabeled_datasets.add_data(data2.data)

###Active Learning

We can specify how many observations we want to query with the first argument of the active_learn function. We can also specify how many observations the user should label before the SVM is re-estimated. 

To demonstrate the learner, we will only go through only 10 reviews, but it is recommended to continue querying until the user is happy with the accuracy, confusion matrix or score.


    active_learner.active_learn(10, num_to_label_at_each_iteration=10)

    labeled 0 out of 10
    -------------------------
    movie reviewers have an obligation to see the good , the bad , and the despicable . 
    i originally wrote this review for my college newspaper back in '95 , but i wanted to re-write it because not all retro reviews should be about the classics . 
    we need to be warned about some truly awful films , too . 
    this picture was so bad , it inspired the description for my 1/10 rating ( see ratings chart below ) . 
    the only thing saving it from a 0/10 rating is that being able to rent a movie like this is slightly less embarassing than renting a porno . 
    so , it does indeed have some plusses . . . 
    in fairness , elizabeth berkley is certainly worth seeing in the buff . 
    and her ability to whine and irritate us , even while nude , was appropriate in her role as selfish temptress nomi malone . 
    this character is not smart , not interesting , and ( deliberately ? ) 
    far too annoying far too often . 
    like in 1998s " bulworth " , when the movie was over i didn't care one little iota about the main character . 
    at least warren beatty tried to make a statement with his dreary and overrated film , though . 
     " showgirls " is too stupid to make a statement . 
    some people claim that the story is based on the legendary " all about eve " of all things ! 
    if exploitation expert joe eszterhas was half the screenwriter that joseph l . mankiewicz was , he'd be . . . well , 
    he'd be a good screenwriter . 
    instead , this project may have sunk his career . 
    he wrote the ( ha ha ! ) 
    script and the usually reliable paul verhoeven directed . 
    the same team who created a good sex-film in 1992 ( " basic instinct " ) struck out here with their occasional violence , gratuitous x-rated sex scenes , and numerous ( and quite unnecessary ) lesbian overtones . 
    the predictable storyline revolves around nomi streaking into las vegas to make it big as a dancer . 
    after the supposed street-smart young " lady " gets conned out of her suitcase by a slack-jawed yokel in the opening sequence , she befriends a tailor of the glamorous stage production at the " stardust hotel " . 
    nomi doesn't take advantage of this contact to break into the big-time of dirty dancing right away . 
    first , she becomes a lap dancer at a scummy strip club . 
    she sells her hot little wares at " the cheetah " for a short time , turning on the fictitious customers and the actual theatre audience . 
    hey , i never said she wasn't a hot number . . . 
    maybe it's that , her body , which gets her into the big-time when the star of the " stardust hotel " , crystal connors ( gina gershon ) stops into the strip club ( it certainly isn't her brains or pleasant disposition ! ) . 
    crystal requests a private lap dance for her boyfriend ( kyle maclachlan ) . 
    he also happens to be the pleasure-seeking worm who runs the big show . 
    crystal gets nomi onto the " stardust " team and , after an interminable amount of time , nomi " earns " the role of crystal's understudy . 
    the slut then deliberately injures and hospitalizes crystal . 
    it was almost laughable that nomi had the guts to claim that she's " not a whore " . 
    that's a phrase we hear a few different times and it's completely ludicrous . 
    of course she is ! 
    she sells her soul to make it big , but in the end we're supposed to believe that she's a better person than that . 
    we're supposed to root for her . 
    no way ! 
    she's a tramp and a back-stabber who deserved nothing--least of all vindication in the . . . ahem . . . climax . 
    she is not a good person and they take over two hours to explain that the audience should think that she is . 
    the sub-plot with a male dancer ( glenn plummer ) who claims to see talent in nomi is just a gratuitous opportunity to let her dance naked a little bit more . 
    it sure ain't character development ! 
    plummer also appeared in " speed " in '94 in a smaller , yet better role . 
    this sub-plot goes absolutely nowhere except maybe to deliver an unsubtle hint that fornicators should practice birth control . 
    kyle maclachlan must have been promised a big pay-day or his standards have dipped since appearing in " blue velvet " . 
    that picture was weird , but some critics claim it's one of the best movies of the '80s . 
    now ol' kyle can say that he acted in one of the worst of the '90s , too . 
    his sleazy character is important to " showgirls " , but we don't learn anything about him . 
    he uses people to get what he wants , but that only means that he fits in well with the other characters in this movie . 
    is he a villain ? 
    who cares ! 
    ultimately , this movie is as tiresome as it is explicit . 
    everyone must know by now about the soft-core acts of copulation , especially the riotous romp between berkley and maclachlan in his pool . 
    what few reviewers take issue with is how mean-spirited this movie is . 
    everyone's either having sex , exacting revenge , or wishing they were having sex or exacting revenge . 
    it's just too hard to take ( especially for over two hours ) . 
    if we must be exposed to the evils of the vegas world , why couldn't verhoeven have also made a point of highlighting the whispers , grunts , and other sounds during the dance numbers ? 
    it's hard to care about these people if we can't even appreciate what they're capable of doing on-stage . 
    berkley may have a future in hollywood because she can dance and she has a great body . 
    after all , the world of porn is still an active , dishonourable profession . 
    perhaps berkley could join their ranks and leave the real acting to pauly shore and cindy crawford . 
    oops , they're bad actors , too . 
    well , at least , they're not selfish and contemptable like good ol' slutty nomi malone . 
    useless trivia--ironically , elizabeth berkley played virtuous and " holier than thou " jessie on the teeny-bopper tv show , " saved by the bell " , before breaking onto the big screen . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 0
    -------------------------
    -------------------------
    i didn't realize how apt the name of this movie was until i called the mpaa ( the motion picture association of america - the folks who decide what's g , nc 17 , pg , r or x ) to ask why the preview was rated r . so that we can make some sense of their response , let me tell you about the movie itself . 
     " the celluloid closet " is a documentary about how homosexuality has been portrayed in the movies over the past several decades . 
    it's brilliant , funny , naughty and extremely poignant . 
    it tore at my heart to watch a gifted lesbian screenwriter explain that , as a rule , gay audiences hunger for any hint of homosexuality on screen . 
    regardless of how veiled - or how sordid - the presence of a gay or lesbian person allows others to lessen their sense of isolation and makes them feel as if they're not quite so invisible as america seems to want them to be . 
    the movie itself is rated r - and for good reason . 
    it contains scenes of bloody , violent gay bashing and graphic , uninhibited , sex . 
    as with any movie , i appreciate knowing about these things ahead of time , so i can decide for myself whether to see the movie with a friend , a date , my 11 year old niece , alone or not at all . 
    but , that's the movie . 
    now back to the preview . 
    prior to this film being theatrically released ( it was originally filmed as a documentary for hbo ) i had seen the coming attractions trailer for it at least six times . 
    there was no nudity , no violence , no bad language , nothing that i could see that would be offensive or inappropriate for a general audience ( okay , whoopi goldberg did refer to someone " boning " someone , but the last i knew that wasn't one of the seven words you can't say on tv ) . 
    except for a scene of two fully clothed men kissing . 
    hmmmmm . 
    when i inquired about the rating on the trailer , a very nice woman at the mpaa quoted from " the handbook " that a trailer approved for all audiences could contain " no homosexuality or lesbianism and no going down on someone . " 
    hello ? i was in the office and it was the middle of the day . 
    bravely , i pursued . 
     " i've seen that trailer , oh . . . 
    probably half a dozen times , " i gulped . . . 
     " and i don't remember that scene . " 
     " well , " she chirped . 
     " it's there . 
    our little eyes are trained to see that . " 
    no really . 
    in the words of dave barry , " i am not making this up . " 
    they are " trained " to " see that ? " 
    when someone who was shocked at the rating the first time and made a note to watch it carefully the following five times or so managed to let it slip past her ? 
    gosh , i certainly don't mean to question the mpaa , or " the handbook " . 
    i would , however , like to suggest that it's they who are in the closet on this one . 
    and the light ain't good in there . 
    but , having seen " the celluloid closet , " and being one of a handful of straight people involved in a primarily gay and lesbian weekly bible study ( email me and i'll give you the details ) , none of this was any big surprise . 
    the point of the movie was that homosexuality , even in the politically correct 90s , is ridiculously perceived as a threat to a mostly heterosexual society . 
    a point well made in this candid and honest film . 
    now , i could go off on the mpaa's ruling that a trailer must contain " no homosexuality or lesbianism " and ask how that is defined , particularly in light of some of the things , both sexual and non-sexual , that i've watched straight people do in trailers . 
    i just don't feel the need to go there , because it seems so obvious . 
    i'll instead suggest that the mpaa re-evaluated their evaluation criteria . 
    let the ratings reflect not subject content , like " sex " and " violence " . 
    let them reflect attitude content . 
    in the future , i'd be interested in knowing whether the movie is rated d for disrespectful or s for stereotyped . 
    then i'd truly be able to make an informed decision about how i spent my time . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 0
    -------------------------
    -------------------------
    my first press screening of 1998 and already i've gotten a prime candidate for my worst ten of the year list . 
    what an auspicious beginning ! 
    welcome to the dog days of winter when the only film openings of merit are those oscar contenders that the studios opened in late december in new york and l . a . and which are just now beginning to appear elsewhere . 
    firestorm , the directorial debut of dances with wolves's academy award winning cinematographer dean semler , is the first of the new year's crop of movies . 
    as our story opens , the movie pretentiously informs us that of the tens of thousands of firefighters only 400 are " smokejumpers . " 
    we then cut to a plane load of smoke jumping cowboys and one cowgirl , where one of the gung-ho guys is taking a romance quiz from " cosmopolitan . " 
    having the time of their lives , they then jump into the middle of a burning forest . 
    when , even in the beginning , the director can't get the small parts right , you can sense the movie is in trouble . 
    with the noisy fire roaring all about them and with the trapped people huddled near their gasoline-filled cars , smokejumper monica ( christianne hirt ) tells them to get away from their soon-to-explode vehicles . 
    not bothering to shout nor even get close to them , she announces her warning without raising her voice much or approaching the people . 
    miraculously , they manage to hear her and move away . 
    in a movie that specializes in cheap shots , the camera locates the proverbial young girl trapped in a nearby burning building . 
    as it does throughout , overly dramatic cinematographer stephen f . windon from the postman uses extremely fast zooms right down to the endangered girl's face . 
    our show's two heroes , the crew's chief , wynt perkins , played laconically by scott glenn , and his second-in-command , jesse graves , played by howie long in a weak attempt to be the next steven seagal , enter the burning house looking for the little girl . 
    in a panic they have difficulty in locating her before they are engulfed in flames . 
    the manipulative script has her hidden in her own dollhouse . 
    this mawkish show cuts back to monica , who has a life-or-death decision to make . 
    the chopper with the fire-retardant chemicals has only enough to save one group . 
    will it be the large group near the cars or the helpless little girl and monica's two firefighting buddies ? 
    she has only seconds to decide who will be saved . 
    yes , she goes for the majority , but , miracle of miracles , the other three come out alive anyway . 
    not content with a traditional firefighting story , chris soth's screenplay attempts to jazz it up by having william forsythe from palookaville play a vicious killer named randy earl shaye who sets a forest fire so that he can join the crew to put it out and then escape . 
     ( " hoods in the woods , " is what the " ground-pounders " yell out when the convicts are bused in to help them fight the fire . ) 
    along the way , shaye picks up an ornithologist hostage played by suzy amis , who turns out to have been trained in warrior ways by her father , who was a marine drill instructor . 
    most of the highly predictable movie is a long chase in which poor howie long is given one ridiculous stunt after another to look silly performing . 
    he flings a chain saw backwards over his head while riding a speeding motorcycle so that the saw can hit the windshield of the pursuing truck . 
    arguably the low point is when he escapes from a locked burning building by riding a motorcycle conveniently parked inside . 
    using a ramp , he shoots straight out of the top of the building's attic , and when he hits the ground , he just rides off in a cloud of dust . 
    when the film isn't using some stock footage of actual forest fires , the simulated ones look hokey . 
    editor jack hofstra cheapens the action even more by his use of burning flames in scene transitions . 
    the ending , with its sick twists , manages to be even worse than the rest of the movie . 
    perhaps the best that can be said for the picture is the faint praise i heard afterwards in the lobby , " it's not as bad as some of the television sitcoms . " 
    firestorm runs mercifully just 1 : 29 . 
    it is rated r for violence and language and would be acceptable for teenagers . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 0
    -------------------------
    -------------------------
    eddie murphy has had his share of ups and downs during his career . 
    known for his notorious late 80's slump , murphy has still managed to bounce back with a handful of hits in the past few years . 
    with the exception of the dreadful holy man , he appears to be on pace for a full-fledged comeback . 
    life was a great move on the part of murphy and co-star martin lawrence , because it's a great showcase for both actors that never resorts to slap-sticky drivel . 
    director ted demme is smart enough to realize that the two comedians can generate enough genuine laughs on their own , and doesn't insert a distracting plot to back them up . 
    life is , in a sense , one great balancing act with murphy on one end and lawrence on the other . 
    amazingly , the scale never tips in either's favor due to the marvelous chemistry and wonderful contrast that each actor allows the other . 
    as the movie opens , we're introduced to ray gibson ( eddie murphy ) , a two-timing pickpocket who schmoozes his way into a club . 
    there he meets a successful businessman named claude banks ( martin lawrence ) . 
    somehow , after multiple contrivances , the mismatched pair find themselves on their way to mississippi on a moonshine run . 
    when all is said and done , ray and claude have been framed for a murder that was actually committed by the town sheriff . 
    hence the setting of life : mississippi state prison , where the main characters come to realize their unlikely friendship is important , and become set on finding an fool-proof escape plan . 
    the film takes us from the 30's all the way to the 90's , presenting a difficult task in showing how the aging process affects ray and claude . 
    luckily , rick baker handles the makeup effects of the two actors in a fantastic , academy award caliber manner . 
    not only do we believe the characters look as if they're 90 years old , but they sound like it , too . 
    murphy and lawrence are completely convincing in the lead roles , even as crotchety old cons bickering over a game of cards . 
    this is just one of the pleasant surprises that the film has tucked up it's sleeve . 
    while the ads are marketing life as a straight arrow comedy , there is a hefty amount of dramatic material hidden at it's core . 
    but the comedic aspects work wonderfully , wisely drawing strength from the talents of the two stars . 
    the movie is more of a comedy than it is a drama , but in both senses , it's an overwhelming delight . 
    i could say a few bad things about the movie , but i don't want to . 
    it's such a nice surprise , such a great vehicle for eddie murphy and martin lawrence , that it warrants a huge smile as the credits begin to roll . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 1
    -------------------------
    -------------------------
    charlie sheen stars as zane , a radio astronomer who listens for sounds from other lifeforms . 
    when he finally gets one , his boss destroys the tape and fires him . 
    naturally , zane is not ready to give up , and he comes up with an ingenious way to do this himself . 
    he is aided by a young neighborhood kid and they discover that the sound is coming from mexico . 
    so zane goes down there to investigate , and runs into a lady studying why the temperature of the earth has dangerously risen so suddenly . 
    zane is having marital problems at the time , and an offer by her to spend the night with him is very tempting . 
    hearing charlie sheen deliver the line , " i guess there is something to be said for celibacy " is the funniest thing i have ever heard in a movie since matthew broderick discussed asexual reproduction in wargames . 
    this is just the setup , and i don't want to give too much away , because a large part of the movies fun is the surprises . 
    charlie sheen , who has had a rocky career as of late , is in top form here . 
    he is funny , serious , and determined to accomplish his goal . 
    sheen's absolutely terrific performance is another big plus to this movie . 
    the story is ingeniously devised by twohy , who also wrote and directed the equally clever cable movie grand tour : disaster in time . 
    the films major flaw is a very slow pace , and not much happens in the earlygoings . 
    viewers may be growing restless for a while , but trust me if you stick around and keep your head in it , you will have a good time . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 1
    -------------------------
    -------------------------
     " remember what the mpaa says : horrific and deplorable violence is ok as long as you don't say any naughty words . " 
    featuring the voice talents of matt stone , trey parker , mary kay bergman . 
    rated r . 
    filmmakers jump on real-life controversies faster than austin powers on felicity shagwell . 
    the debate on whether cinema is to blame for teenagers turning into hoodlums has only begun to heat up , and already someone's made a movie about it , and it's not , thank god , a tragic account of a family ripped apart by the effect violence in films had on a teenage boy . 
    instead we get a sharp , biting satire that takes no prisoners and leaves no conservative point of view unscathed . 
    based on a popular ( and controversial ) cable tv show , south park : bigger longer and uncut can finally break loose of the shackles placed on the show by television restrictions and take bad taste to brand new heights . 
    the movie is about a group of kids who sneak into a canadian r-rated movie and learn some naughty words . 
    when they exhibit their new knowledge to their moms , they decide to " blame canada , " wage war against the neighbor country and execute " terrence and phillip , " the flatulent actors in the obscene film . 
    the kids form an alliance they name " la resistance " ( with the accent on the third syllable of " resistance " ) to save their favorite thespians , in a hilarious spoof of ( tribute to ? ) 
    les mis ? rables . 
    in a subplot , one of the characters ( kenny , who else ? ) 
    dies and goes to hell where he meets satan . 
    satan and saddam hussein are lovers , you see . 
    satan is a benevolent soul , while our favorite eastern ruler can only think about sex . 
    apparently , too , if terrence and phillip are executed it will be the final sign of the apocalypse and satan can emerge from the deepest bowels of the underground kingdom to rule the earth . 
    aside from being a brilliant satire , south park is also an all-stops-out musical , with unforgettable numbers like " shut your f * * * ing face uncle f * * * er " and " cartman's mom is a big fat bitch " . 
    almost invariably it's funny stuff : often juvenile but always funny . 
    the same can be said for the rest of the movie : it's intelligent but delivered in a sophomoric manner ( i . e . 
    toilet humor , endless profanity , etc . ) . 
    not that there's anything wrong with that : vulgarity , when done right , is my bag , baby . 
    trey parker and matt stone , the twenty-somethings behind the film and the show , paint a bulls-eye on the motion picture association of america and proceed to be the first to start trying to hit it . 
    cruelly mocking the fact that the mpaa's rating system will allegedly tolerate grotesque violence as long as obscenities aren't uttered , the auteurs throw some nasty one-liner insults their way . 
    the ratings-a-plenty association isn't the only target of this unsparing banter : people who favor censoring movies over gun control are equally fair game , with the " blame canada " plot being a not-so-cheap shot at them . 
    the distinctively low-tech " cardboard " animation is oddly effective , even more so than the state of the art " deep canvas " technique aptly demonstrated in the recent tarzan . 
    it's more pleasant to look at , less intimidating up on the screen , and most importantly it doesn't detract from the film's concept as much as disney's admirably awe-inspiring work does . 
    the show's popularity has been waning as of late , and perhaps this movie is just the thing to boost its ratings . 
    perhaps not . 
    having seen the show on numerous occasions , i can say that it's not nearly as smart or as funny as this movie . 
    the series may be better off simply continuing on the big screen every couple years . 
    parker and stone have outdone themselves to the point where i am forced to ask : must the show go on ? 
     ? 1999 eugene novikov&#137 ; 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 1
    -------------------------
    -------------------------
    what are the warning signs of a * terrible * movie ? 
    making it's debut at the dollar theater ? 
    locally , chairman of the board did just that . 
    having the annoying prop comic scott thompson ( better known as carrot top ) in the lead role ? 
    chairman of the board , once again . 
    how about an overly exhausted , paper thin plot approached with utter incompetence ? 
    did somebody say chairman of the board ? 
    that's right , carrot top's long dreaded major motion picture debut ( at least for a starring role ) is poking up in a handful of theaters across the country . 
    chairman of the board stars the obnoxious , wannabe-zany king of redheaded standup comics as a lazy but creative , inventive but uneventful generation x- er named edison . 
    living with a pair of surfer dudes in a small , rented house , edison bounces from job to job , always squandering away the money on his eccentric ( to say the least ) inventions and ignoring crucial responsibilities such as rent . 
    this has the crabby landlady , ms . krubavitch ( estelle harris , best known as george constanza's mother on " seinfeld " ) , threatening an eviction if past due expenses aren't furnished post haste . 
    as luck would have it , edison soon meets armand mcmillan ( jack warden ) , an old surfer dude who just so happens to be president of the multi-million dollar mcmillan industries . 
    sharing a passion for more than just riding waves , armand is deeply impacted by the young inventor's notebook of dreams and ideas , and when the old man dies soon afterward , edison learns he is named a benefactor in armand's will . 
    predictably , edison acquires the entire corporation and has to maintain productivity with absolutely no knowledge of the business world . 
    predictably , there is a bitter nephew ( larry miller ) whose lesser inheritance fuels resentment that will lead to an elaborate sabotage plot . 
    predictably , there is an attractive employee ( courtney thorne-smith ) whose initial repulsion will transform into love for our doofy protagonist . 
    predictably , the man who knows nothing will fight against the odds and give the company it's most profitable and successful turnaround ever , all because he ran things by common sense and not greed . 
    it's as though writers turi meyer , al septien , and alex zamm ( meyer and septien also wrote leprechaun 2 together ! ) 
    pulled a plot out of a hat and worked carrot top into it . 
    the jokes , the " surprises " , the developments - all of them run such a predictable path , it may only be carrot top's signature brazen red hairdo that sets this one apart from the myriad of similar films . 
    a movie this bad speaks for itself . 
    what's left to say when every element the movie possesses is a shameful retread of movies past ? 
    the script is 100% recycled , the direction is hokey , and the acting is absolutely horrible . 
    it is only thorne-smith who seems to take her job seriously , an accomplishment which surely deserves the medal of honor . 
    she certainly went beyond the call of duty - she has to kiss carrot top ! ! ! ! ! ! 
     ( barf bag , please ! ) 
    movies like this give the audience nothing to do but ponder just how many synonyms for " bad " there really are . 
    chairman of the board , without a doubt , deserves each and every one . 
    the only way this won't end up on everybody's " bottom ten of the year " list , is if they were lucky enough never to have seen it . 
    just because you can't miss his outlandish fiery mane , don't skimp on avoiding this abhorrent feature . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 0
    -------------------------
    -------------------------
    come on hollywood , surprise me . 
    stop giving us these poorly written thrillers with banal dialogue , sketchy characters and plots as predictable as the sunset . 
    the always watchable morgan freeman plays a detective who becomes personally involved in a case involving missing girls . 
    personal , because his niece is one of the victims . 
    it's a slobbering psychopath , of course , but this time there's a twist . 
    freeman notes that each of the young women who've disappeared are all strong willed , assertive , and more successful in their careers than the average girl . 
    we soon learn that the guy calls himself casanova , whose aim is to " dominate " these modern gals by imprisoning them in some dungeon and keeping them as his personal harem . 
    anyway , one of the women manages to escape ( ashly judd ) and teams up with freeman to . . . well , 
    you know the rest . 
    a brief glance at the plot to silence of the lambs , with which this film is constantly being compared to by the hype merchants , may suggest lambs also has a fairly predictable story . 
    perhaps , but that film also has superbly drawn characters and smart dialogue . 
    so lets not insult a great movie by taking the comparisons any further , okay ? 
    and as for comparisons to seven . . . oh 
    please ! 
    kiss the girls is based on the novel by james patterson and written for the screen by one david klass . 
    maybe the novel was a stinker to start with , but whatever the case , it's the writing that's clearly at fault here . 
    one , these characters have very little to say that's engaging or interesting . 
    two , the script has no sense of humour . 
    three , while the notion of a psycho's victims being smart , successful women is an interesting twist , the execution isn't even half as good as the idea . 
    thus , ashly judd comes across all out of focus , instead of being the heart of the story . 
    we can't feel her rage . 
    some atrociously written casual exchanges between her and several male characters are supposed to remind us that she's the no-bullshit 90's type , but these conversations barely register . 
    and as for freeman , kiss the girls is his second dog in a year : first chain reaction , now this . 
    for an actor of his calibre , this is most worrying . 
    young director gary fleder scored a hit a few years back with his quirky pulp fiction-esque things to do in denver when your dead . 
    but what can he do with material as resolutely mediocre as this ? 
    not much , and you can hardly blame him . 
    there's a few well-staged chase scenes through the forest where the camera whirls , dives and jumps , and the effect is startling . 
    but the script is beyond rescue . 
    what hurts most is that hollywood continues to get away with serving up this tripe , safe in the knowledge that jaded audiences will lap it up . 
    complacency rules : it's been so long since we saw a mainstream american thriller that delivered juicy characters , real surprises and consistently sharp dialogue . 
    the only consolation for this viewer is that my ticket to the movie was a freebie , positive proof that the best things in life aren't free . . . . 
     . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 1
    -------------------------
    -------------------------
    ahh yes . 
    the teenage romance . 
    an attractive young cast is pitted into an unlikely scenario that could only happen in the movies , and in the end , the guy always gets the girl . 
    and with the arrival of the breakout hit `she's all that' last year ( followed by a long catalogue of imitators including `10 things i hate about you' and `drive me crazy' ) , the genre previously on life support is once again a hot commodity . 
    along now comes `down to you' . 
    the folks at miramax are obviously trying to capitalize on the rabid `she's all that' craze with their latest project , which has the studdly freddie prinze jr . attached and all . 
    only `down to you' doesn't have the ? unlikely scenario' mentioned above . 
    it is an extraordinarily ordinary romance - a dull , unattractive teen comedy that sticks to the boring game plan that we're accustomed to . 
    this is the kind of romance that only giggly 12-year old girls will find convincing . 
    in strictly textbook fashion , college sophomore and aspiring chef al ( prinze jr . ) meets freshman artist imogen ( julia stiles ) . 
    they hit it off like a couple can only in the movies . 
    from here , it's the standard boy-meets-girl , boy-loses-girl , boy-drinks-entire-bottle-of-shampoo-and-may-or-may-not-get-girl-back story . 
    the plot is conveniently assembled to suit the requirements of the main characters , who are frequently taking part in activities that . . . . 
    well , only happen in the movies . 
    fortunately , the cast of `down to you' has a certain appeal . 
    freddie prinze jr . and julia stiles are an adorable couple , and when on screen together , they radiate the sort of warmth and charisma that the movie should have centered around . 
    zak orth , as the newly realized porn star monk , shows an unmistakable flair with handling all of the film's intelligent dialogue . 
    rounding out the impressive ensemble of young talent are shawn hatosy ( `the faculty' ) , selma blair ( `cruel intentions' ) and ashton kutcher ( tv's `that 70's shows' ) . 
    even the fonz himself - henry winkler , the epitome of teenage angst and nostalgia - has a welcome role as al's dad , the host of the popular cooking program `chef ray' . 
    maybe the concept of `down to you' looked good on paper to draw such a crowd . 
    as a feature film , however , the finished product is bland and tasteless fluff with only an occasional whiff of cuteness to keep the gears from stopping entirely . 
    perhaps worst of all , `down to you' is not funny . 
    the jokes are drawn from obvious sources and the resulting humor is banal and uninspired . 
    the characters on screen , often laughing at each other's goofy/embarrassing antics , seemed far more amused than the audience . 
    even the giggly 12-year old girls had grown restless toward the end as they waited impatiently for the formula to run it's course . 
    the one mildly clever segment featured winkler and prinze jr . in a fantasy sequence called `cooks' , a `cops' take-off in which the father and son would storm houses and cook a decent meal for ? needy' families ( with the assistance of a fully-armed swat team , of course ) . 
    when this is the highlight reel , you know the remaining film could leave something to be desired . 
    in order to make a teen comedy work , you've got to have characters that show something by way of depth and identity . 
    `10 things i hate about you' actually featured characters who were more than walking flashcards , and the result was refreshing . 
    no such luck with `down to you' . 
    it's just a textbook romance where , despite absurd circumstances , everything is bound to work out in the end . 
    and at the end of the whole clich ? d ordeal , the nicest thing you could possibly say would be ` . . . . 
    only in the movies' . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 0
    -------------------------
    -------------------------
    you've got to think twice before you go see a movie with a title like maximum risk . 
    the title is generic . 
    it's meaningless . 
    and i can't believe it's good business . 
    when you pick up the phone and dial 777-film , how in the world are you supposed to remember that the new jean-claude van damme movie , the one that looked kind of cool in the trailers and is directed by some chinese hot shot , is called , geez , maximum risk ? 
    yuck . 
    the movie itself deserves your attention . 
    for sweet bloody thrills , this one beats the summer blockbuster competition hands-down . 
    only mission : impossible came close to delivering as skillful a thriller , and i'll give maximum risk the edge simply because it's not as slick as the tom cruise picture , and therefore more gratifying in its execution . 
    much to my surprise , van damme continues to develop as a pleasant , unpretentious action hero . 
    his track record isn't as solid as schwarzenegger's , but he's a hell of a lot more adventurous than arnold . 
    in 1993 , van damme worked with hong kong's premier hardcore action director , john woo , on a fairly lame movie called hard target . 
     ( if you can find a bootleg copy , woo's radically different director's cut is much better than what was eventually released . ) 
    this follow-up is directed by ringo lam ( city on fire , full contact ) , whose hong kong films are distinguished action pictures that have consistently played second fiddle to woo's more operatic offerings . 
    the surprise here is that maximum risk is a more effective hollywood action flick than either hard target or woo's subsequent broken arrow . 
    here's the rundown . 
    van damme plays a french cop named alain moreau who is shaken when a policeman friend ( jean-hugues anglade ) finds a corpse that's alain's exact double . 
    turns out alain was separated at birth from his twin brother , michael , who has been killed by some russian heavies ( and some strangely american looking cops ) . 
    alain does some investigating . 
    he finds that michael had booked a flight to new york city , and received a message from someone there named alex bohemia . 
    assuming michael's identity , alain flies to new york and gets tangled up with michael's girlfriend ( natasha henstridge ) , the fbi , and the russian mob in little odessa . 
    that's as much as you need to know . 
    the story is adequate , but not overly involving -- and the major plot points are basically explained to you twice , just in case you go out for popcorn at the wrong moment . 
    there's a love story , too , but i didn't find it terribly convincing , partly because ex-model henstridge is too high-strung in her high-profile debut ( she had precious few lines as the alien ice queen in species ) . 
    she's great to look at , and she can certainly read a line , but what she does here can't really be described as " acting . " 
    of course , " acting " isn't really what she was hired for ( i lost track of whether her shirt comes off more often than van damme's ) . 
    the movie is exceptionally violent , bordering on gratuity . 
     ( keep that in mind when planning your date . ) 
    the stunts are spectacular , and the fact that you can spot van damme's stand-in makes his work no less impressive . 
    there's only so much you can do with a car wreck , but this movie makes crisp , effective use of pile-ups in a handful of frenzied destruction derbies . 
    and lam has a surprising , innovative sense of exactly where the camera should go to catch any bit of action . 
    the main difference between lam and woo , i believe , is that while woo relies on sheer spectacle to gas up his action show pieces , lam has figured out more about using cinematic space and double-barreled points of view to make things run . 
    don't get me wrong -- when chow yun-fat soars through space pumping bullets out of two pistols and chewing on a toothpick with glass and confetti littering the air around him in woo's hard-boiled , it's an amazing moment . 
    but it's a moment that's hard to reproduce in hollywood . 
     ( for one thing , hollywood doesn't yet have chow yun-fat ! ) 
    while woo's hollywood movies look like the work of a talented upstart , maximum risk is a surprisingly confident picture . 
    the very first shot of the film is an awkward overhead view of a chase through the streets of a european city , but lam's use of odd camera angles becomes more efficient later on . 
    the film editing is a particularly savvy complement to lam's shooting style , accentuating rather than amplifying the action . 
    the performances could have used some fine tuning ( in particular , there's an annoying , overwritten manhattan cab driver in the early scenes who should have been toned down or jettisoned completely ) , and the movie doesn't always overcome the limitations of its genre . 
    the story is a little mundane , although there are some effective moments involving van damme's unrequited feelings toward the brother he never knew he had . 
    but it's not often that hollywood cranks out a truly satisfying action picture , and it's doubly surprising that this one should come with a mere whisper of publicity . 
    van damme fans should treat themselves to what may well be the man's best movie , and international action buffs will no doubt savor this flavorful hong kong/hollywood hybrid . 
    
    Please enter label for the above point: 
    Please choose from [1, 0]
    Label: 1
    -------------------------
    training model(s) on 12 instances
    finding optimal C, gamma parameters...
    C:0.0625; gamma:0.0625
    positives found during learning: 0
    negatives found during learning: 0
    evaluating learner over 50 instances.
    confusion matrix:
    {'fp': 24, 'tn': 4, 'fn': 3, 'tp': 19}
    done.
    models rebuilt with 12 labeled examples
    active learning loop completed; models rebuilt.


While the accuracy has gone down from only using the initial two observations (we are now using 12), the gamma parameter increased allowing less constraint on the model. An obvious solution would be to query more observations until the accuracy is at a point which we are satisfied with. 

We can get more information of the learners performance on the test set by printing the following.


    active_learner.test_results




    {'accuracy': 0.46,
     'confusion_matrix': {'fn': 3, 'fp': 24, 'tn': 4, 'tp': 19},
     'npos': 0,
     'probabilities': 'This is not a probability model',
     'scores': [{(0, 1): -7.258369579231372e-06, (1, 0): 7.258369579231372e-06},
      {(0, 1): -1.8447004410768186e-05, (1, 0): 1.8447004410768186e-05},
      {(0, 1): -1.1893830628426083e-05, (1, 0): 1.1893830628426083e-05},
      {(0, 1): -2.022838119539183e-05, (1, 0): 2.022838119539183e-05},
      {(0, 1): -1.5457828825629627e-05, (1, 0): 1.5457828825629627e-05},
      {(0, 1): -2.2760783023371978e-05, (1, 0): 2.2760783023371978e-05},
      {(0, 1): 1.287679805166142e-06, (1, 0): -1.287679805166142e-06},
      {(0, 1): 1.3609718686573291e-05, (1, 0): -1.3609718686573291e-05},
      {(0, 1): -2.191186101362097e-05, (1, 0): 2.191186101362097e-05},
      {(0, 1): -1.6449190352006704e-05, (1, 0): 1.6449190352006704e-05},
      {(0, 1): -1.5668120505407668e-05, (1, 0): 1.5668120505407668e-05},
      {(0, 1): -1.3207894122187203e-05, (1, 0): 1.3207894122187203e-05},
      {(0, 1): -1.019264761661226e-05, (1, 0): 1.019264761661226e-05},
      {(0, 1): -1.3528221674836971e-05, (1, 0): 1.3528221674836971e-05},
      {(0, 1): -1.1400006406731e-05, (1, 0): 1.1400006406731e-05},
      {(0, 1): -1.367811772295946e-05, (1, 0): 1.367811772295946e-05},
      {(0, 1): -1.938090900124867e-05, (1, 0): 1.938090900124867e-05},
      {(0, 1): -7.6226671255688605e-06, (1, 0): 7.6226671255688605e-06},
      {(0, 1): -1.8873541094220625e-05, (1, 0): 1.8873541094220625e-05},
      {(0, 1): -1.3346193606787737e-05, (1, 0): 1.3346193606787737e-05},
      {(0, 1): 3.0849256961035243e-06, (1, 0): -3.0849256961035243e-06},
      {(0, 1): 2.4351711866973935e-05, (1, 0): -2.4351711866973935e-05},
      {(0, 1): 2.0446970599025116e-06, (1, 0): -2.0446970599025116e-06},
      {(0, 1): -2.0837833809864148e-05, (1, 0): 2.0837833809864148e-05},
      {(0, 1): -1.7992350108991806e-05, (1, 0): 1.7992350108991806e-05},
      {(0, 1): -1.6256713830466174e-05, (1, 0): 1.6256713830466174e-05},
      {(0, 1): -1.1325175987689229e-05, (1, 0): 1.1325175987689229e-05},
      {(0, 1): 5.543939215882365e-06, (1, 0): -5.543939215882365e-06},
      {(0, 1): -1.599995107616492e-05, (1, 0): 1.599995107616492e-05},
      {(0, 1): -1.591051751553829e-05, (1, 0): 1.591051751553829e-05},
      {(0, 1): -1.0245036316004763e-05, (1, 0): 1.0245036316004763e-05},
      {(0, 1): -1.5112599620531464e-05, (1, 0): 1.5112599620531464e-05},
      {(0, 1): -1.2009368011114963e-05, (1, 0): 1.2009368011114963e-05},
      {(0, 1): -1.3343981797336868e-05, (1, 0): 1.3343981797336868e-05},
      {(0, 1): -1.8450240669175277e-05, (1, 0): 1.8450240669175277e-05},
      {(0, 1): -1.1728485287472445e-05, (1, 0): 1.1728485287472445e-05},
      {(0, 1): -1.4460259171983347e-05, (1, 0): 1.4460259171983347e-05},
      {(0, 1): -1.6799526212189775e-05, (1, 0): 1.6799526212189775e-05},
      {(0, 1): -8.954300259059189e-06, (1, 0): 8.954300259059189e-06},
      {(0, 1): -1.1775207591213643e-05, (1, 0): 1.1775207591213643e-05},
      {(0, 1): -1.585777440264724e-05, (1, 0): 1.585777440264724e-05},
      {(0, 1): -3.6654952164233423e-06, (1, 0): 3.6654952164233423e-06},
      {(0, 1): -1.2178582750599354e-05, (1, 0): 1.2178582750599354e-05},
      {(0, 1): -1.6926358242436157e-05, (1, 0): 1.6926358242436157e-05},
      {(0, 1): 1.9318841822497934e-05, (1, 0): -1.9318841822497934e-05},
      {(0, 1): -1.3437625458426194e-05, (1, 0): 1.3437625458426194e-05},
      {(0, 1): -1.0534246452982565e-05, (1, 0): 1.0534246452982565e-05},
      {(0, 1): -1.5037474525546324e-05, (1, 0): 1.5037474525546324e-05},
      {(0, 1): -1.9474090619503448e-05, (1, 0): 1.9474090619503448e-05},
      {(0, 1): -4.588036264033257e-06, (1, 0): 4.588036264033257e-06}],
     'sensitivity': 0.8636363636363636}



The scores are the decision values of the support vector machine. The classifier chooses the class based on the sign of this value. If it is positive relative to (0,1), it will predict class 0 (negative) and if it is positive relative to (1,0) it will predict 1 (positive). 

If a probability model was selected to begin with, then probabilities will be reported instead. Note, that this model is more computationally intensive and requires much more training data. 

I hope this tutorial was helpful. Any comments would be very appreciated: paul.soto@upf.edu.
