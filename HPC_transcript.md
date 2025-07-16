CppCon 2017: Carl Cook “When a Microsecond Is an Eternity: High Performance Trading Systems in C++”
https://www.youtube.com/watch/NH1Tta7purM

Cool so yeah, good afternoon everyone.
My name is Carl, I work fora company called Optiver.
I work writing automated trading systems
and I've done that forabout 10 years in total.
So today's talk is just going to be
basically a day in the life of a developer
of high frequency trading systems.
I'm also a member of SG14,
so I try to give a little bit of feedback
from the trading communityinto their standards.
What's important from thelow latency perspective
from trading.
But I'm definitely not a C++ expert.
There's people in this roomwho will know 10 times more
about C++ than me.
It's not necessarily a problem though.
You just need to be reallygood at measurement.
You need to be really good atapplying scientific method.
You need to understand the tools.
If you can do that and you can measure
and you can measure well, you can write
very very fast softwarewithout knowing the standard
inside out.
So I'm gonna give a veryquick overview of what we do.
Won't go into that in much detail,
'cuz that can take an houror two just by itself.
But I will talk aboutthe technical challenges
that we face.
Those two topics are just an introductory,
then we get into the real stuff,
which is the actual techniquesthat I use on a daily basis
for writing fast C++.
Mention some surprises, some war stories,
and then finally I'm gonnadiscuss how to measure C++
and that's probably themost important few slides
of the talk.
It's really all about measurementif you want fast code.
Cool.
Another thing is that I wouldabsolutely love to be able to
discuss absolutely everyoptimization trick out there,
but I can't.
Just not enough time.
I just did a run throughof my slides this morning,
threw away about half of them.
So yeah, this is only a very very brief
sample into optimizationtechniques for low latency.
So low latency, not high frequent,
not trying to process60, 100 frames a second,
but just trying to be very very fast
when you need to be fast.
So what do we do?
Well most market makersare providing liquidity
to the market.
That basically means we'reproviding prices to the market
to say we're willing tobuy this instrument for
a certain amount, we're willing to sell it
for a certain amount more than that.
By instruments I mean a tradable product,
a stock, a future option.
It's just pure coincidencethat there's instruments
on the table over there.
Not those sorts of instruments.
This has some challenges.
You need to be moving yourprices quite regularly.
If you don't move your pricesyou'll miss out on the trade
or even worse you'll getthe trade but it wasn't
what you're looking for 'cuzyou didn't move your price
fast enough and someoneelse has lifted you on that.
So what market makersdo is they try to just
make the differencebetween buying and selling.
So they want to buy astock and sell it again.
Sell it, buy it.
That will hopefully balance itself out.
So market makers aren't quantitative funds
or hedge funds or anything like that.
They're not looking to build a position
with the intention that themarket's going to go up or down,
they're really just trying tomake a small amount of profit
by buying and selling fairly frequently.
Ultimately this alwayscomes down to two things:
buying low, selling high.
Yeah there's some complex maths in there.
Sometimes it gets a little bit complex,
but ultimately that's whatall trading algorithms
come down to.
And success means beingjust slightly faster
than your competitors.
It doesn't need to be by a second,
it could be a picosecond,just as long as you're faster.
I think this is quitea nice picture actually
because it gives you an idea.
No one really remembers whocomes second or third or fourth
in a race right?
And that's kind of thesame thing with electronic
market making.
You really need to bethe guy out in front.
But, safety first.
A lot can go wrong in a veryvery short period of time.
So you need to be able todetect errors and get out,
not the other way around.
You don't want to think about things
then get out of the market,
you want to get out thenfigure out what's gone wrong,
because a lot can gowrong in a few seconds.
That's best automated obviously.
That's a good quote.
Okay, so I'm gonna talkabout the hotpath a lot,
sometimes I'll say fastpath,but it's the same thing.
This is the code which sends a message
from the exchange, decodesit, figures out that
this is an interesting event,this is a price change,
this is a trade, this issomething, a market open event.
Executes an autotrading strategy,
decides it wants to trade,runs the risk checks,
makes sure that that's okay,
and then finally sendsan order to the exchange
to say I'd like to trade orI'd like to change my price
or something like that.
That's the hotpath.
Characteristics of thehotpath are quite interesting.
Probably one to five percentof the total code base
and it gets executedvery very infrequently.
Maybe .01% of the time you'llactually have that hotpath
fully execute.
Now that's at odds withthe operating system,
the hardware, the network,basically everything in the room
is at odds with tryingto execute a hotpath.
Not very frequently, but whenit executes very very fast
very little latency.
So even networks for exampleare based on fairness,
which makes sense.
They want to cache upbytes until there's enough
for the packet to be fully sent,
then the packets sent andmake sure that no one sender
is saturating the network.
The same with operatingsystems, they're all about
multitasking and cooperative multitasking.
These are things thatyou don't actually want
from a low latency perspective.
One final point: jitter.
You do want to be fast most of the time,
but sometimes you're slow.
So if you're fast four timesout of five, that's great,
but if that fifth time you're so slow
that you end up with a badtrade because you didn't
move your price or you missed out,
well if you miss outyou miss an opportunity,
but if you get hit on abad order or bad trade,
that can be quite expensive.
So it's very frustrating.
You can have code whichruns faster on average
a lower median, but thestandard deviation is too wide
and that's actually not a good thing.
You prefer a tighter deviationthan an actual faster median.
So where does C++ come into this?
Well it's the language ofchoice within trading companies
and that's no coincidence.
It's low cost.
You get basically zero cost abstractions.
You can get your hands onthe hardware reasonably,
but it's still a relativelyabstract language.
That's great, that's what we want.
But there's some catches.
Your compiler, and compilerversion, build link flags,
different libraries that you're using,
the machine architecture,these are all going
to impact your code.
And they will unfortunatelyhave interactions
with each other as well.
So you really need tonot just know C++ well,
but figure out what thecompiler is actually generating.
Anyone know an app for that?
Very very useful.
No coincidence that thiscomes from a trading company.
You'll learn more aboutthat on Friday I believe.
Tools like this are great.
You can change the compilerflags, see what happens,
have a look at what theactual resulting assembly is,
change your versions,compare client to GCC,
it's great.
See which version of thecompiler actually made a change
to your code if you notice a regression.
Now I should point out as well,
I don't know how to write assembly code.
But I do know how to read alittle bit of assembly code
and that's great.
So from a performance point of view,
I know that a call or a jumpis going to be expensive.
I know that more machine codeis probably more expensive
than less machine code.
I know that if I have fewerjumps, more conditional moves,
and things like that, we'reonto a good thing here.
Now just to talk a tinybit about system tuning.
This is a C++ conference,so I can't go into this
in too much detail, but it'sworth just pointing this out.
So you can have exactly the same hardware,
exactly the same operating system,
the same binary, the same everything,
but put them on a slightlydifferent configuration
from a BIOS point of viewor a Linux kernel tuning
point of view, you get different results.
So the blue server
is a machine that is notrunning hyper threading.
The orange server is a machine
that is running hyper threading.
This is sorting different sizevectors of random integers.
Why is the hyper threading ...
Oh this is a single threaded (mumbles)
running on one CPU.
Why would the hyperthreading slow us down here?
What's common between thetwo threads on a core?
If you're hyper threading.
Cache. Exactly.
So you're sharing the same cache.
So your cache either goesdown to 50% or more likely
it goes down to zero becausethere's something else running
on the second thread.
Hyper threading is great,
I'm not here to bag on hyper threading,
I use it at home on my desktop, why not?
It turns four cores into eight.
But for a productionserver running low latency
trading software, thingslike that will catch you up.
You don't need to knowabout this from a C++
development point ofview, but you better hope
that there's someone in your company
who can take care of that for you,
otherwise you've alreadylost unfortunately.
So how fast is fast?
The Burj Khalifa in Dubai 2,700 feet tall.
Very tall building, tallestin the world at the moment.
Speed of light coincidentally,is around about one foot
per nanosecond.
So if we developed a autotrading system
that ran end to end, wire towire, from seeing the packet in
to performing of the logicand sending a packet back out
to the exchange in say twoand a half microseconds,
well that's actually lesstime than it takes for light
to travel from the top ofthe spire of this building
to the ground.
That's not a lot of time.
So when you go oh well it's alright,
the machine might be swappinga little bit at the moment,
or we'll go out to mainmemory or it doesn't matter
that someone else is usingour cache at the moment,
well it does.
All bets are off at this point.
You have lost the game at this point.
There's such little time toactually execute your code.
Okay so,
that's the introductions done.
I've got about 10 to 15 coding examples
of just low latency techniquesthat have actually helped me
in production make the code faster.
This isn't just micro benchmarking,
it's and yeah this appears faster,
this is actual code that'sbeen tested in production
and it's made a serious difference.
Most of these will be
somewhat obvious tosome people in the room,
other people maybe not.
So don't be too worried ifsome things you already know.
Hopefully there's a fewsurprises in there for you still.
So this gave me about a100, 200 nanosecond speed up
a couple months ago when I did it,
which was removing the code on the left
and replacing it withthe code on the right.
Why would this be faster?
Mhm.
So indeed, the compiler canoptimize the hotpath better.
What else?
Yes, two answers at onceand they were both super
and they were both correct.
One said cache production,one said branch production
I think.
Well cache and branching, yeah so,
branch production, there's fewer branches
on the right hand side forthe hardware branch predictor
to deal with.
And cache, this is good,
on the right is it betterfor your instruction cache
or your data cache?
Both?
I don't think he said both but (laughs)
I claim it's both.
Yeah, you've got fewer instructions,
less pressure on the instruction cache.
And also who knows what theseerror handling functions
are actually doing?
Maybe they're tramplingyour data cache as well.
Whereas now we've only got one integer
that we're actually reading and writing to
and machine architecturesare incredibly good
at testing if an integeris zero or nonzero.
Does this seem obvious or ... ?
After I did it, it was likeoh why am I not doing this
all the time?
But yeah, just things like this can really
make a difference.
Hi.
Yeah yeah yeah.
So where is the error checking code?
So the error checkingcode is behind the scenes.
It's miles away.
But if there's any flag thathas been set by anything
before heading thishotpath, we'll set a flag
to say do not go any further.
Yeah.
Then in this handleerror, that can figure out
what the actual error is, deal with it.
Does that kind of answer your question?
Yeah.
Okay another quite simple example.
I see this quite a bit,which is template-based
configuration.
Well, the lack oftemplate-based configuration.
Sometimes you don't knoweverything at compile time,
so maybe you can read from configuration
and based on that you'regoing to instantiate
one type of class foranother type of class,
but you don't know whatclass you're actually going
to instantiate.
Most people will usevirtual functions for this.
Virtual functions are absolutely fine,
if you don't know what classyou're going to instantiate.
If you don't know allof the possible classes.
If you know the complete set of things
that may be instantiated,
then you don't reallyneed virtual functions.
You can just use templates for this.
And again this is a very simple example,
but I think it's oftenoverlooked and I see virtual
calls going all the way throughcode and into the hotpath
and it is slow and it is not required.
You see this all the time in the ASDL
and it works really really well.
So here we've got two classes,
and both send orders; ordersender one, order sender two,
A and B.
And we've got an order manager which uses
one of those two types.
We don't know which one andit doesn't know which one,
it's templated at this point in time.
Now there is a virtualfunction core in there,
the mail loop, but that'sjust to start the program
so that virtual call is gonna disappear
before you know about it.
But the actual calls to this order sender
and nonvirtual concrete time.
How do you hook this up?
Just use the factory function, yeah?
Look at your configurationand then instantiate
the correct type, passit in, and you're done.
No need for virtuals.
Again, this is not a ...
I won't win any awards with this slide,
but so many times I'venoticed that people miss these
sorts of things.
You know everything at compile time,
you know the complete setof things that can happen,
use that informationto make your code fast.
No virtuals, fastercode, more compact code,
more chance for the optimizer to optimize,
less pressure on the instruction cache,
less pressure on the data cache.
It's a good thing.
And the next one, Lambda functions,
this will not give you a speed up.
This slide will not give you a speed up,
but it's a nice example of howC++ can be fast and powerful,
but still relatively expressive as well.
So on the left we've got afunction called send message,
takes in a lambda, prepares a message,
invokes the lambda, sends the message.
On the right is the actual usage.
So here we're calling it send message,
we need to pass it a lambda,so here's the lambda,
the lambda takes a message
and then it just populates this.
Now, this could very well be as follows:
send message, on theleft, the prepare message
isn't allocating memoryor anything like that,
it's just getting thenetwork cards buffer,
via DMA, and returning that.
The lambda will definitelyget inlined at two moves,
which is the slow level as you can get.
Send could be as simpleas flipping one single bit
on the network card to tellthe network card to send.
So that's very very low level,very powerful, very fast,
and in fact, the entirefunction there on the left,
send message, will probablyget inlined as well,
depending on what you're doing with it.
So it's very very nice, very very fast,
but still relatively high level language.
So you can see why C++ is somewhat
the language of choice here.
Quick show of hands, whoknows that memory allocation
is slow?
Good.
So that should be no surprise to you.
Use a pool of objects.
I don't care how it's done,you can use an allocator,
you can do it by hand, I don't mind.
But definitely try toreuse and recycle objects.
Be careful with distractingobjects as well,
this can get expensive.
Underneath all of thiswill be a call to free.
Free's actually relatively expensive.
If you go and have a look through glibc,
it's about 400 lines of code.
Yeah you're not going to hitevery single line of code
each time, but even theact of deleting objects
can be relativelyexpensive, so do watch that.
Okay, exceptions.
Are exceptions slow?
If you don't drop into them.
And throw, yeah.
So absolutely if you havean exception that's thrown,
from my measurementit's one or two micros,
but as far as I can tell,and I've measured this a lot,
I've really tried to measure this
in many many different ways.
Exceptions claims to be azero cost thing as long as
they don't throw.
As far as I can tell,that is absolutely true.
So don't be afraid of exceptions.
Of course, if they throw,it's gonna be expensive,
but if an exception is thrownyou've got a problem anyway.
You're not going to besending an order anymore.
So don't use exceptionsfor control flow, yeah?
One, it's gonna be slow, andtwo, your code's gonna look
really really bad.
This is a little bitsimilar to a previous slide.
You don't really need if statements.
Particularly in your hotpath.
Branching is bad, branching is expensive,
and you can remove quite a bit of it.
So here we've got a branchingapproach of running a strategy
where we calculate a price,check some risk limits,
and then send the order.
And if you look at the implementation
of one of these methods, here'sour branching at the side,
and so by then we want toask for a bit of a discount,
otherwise it must be asale, so we're asking
for a little bit of a premium.
That's pretty simple.
This should run relatively fast.
But we know at compile time,
it can either only be a buy or a sell,
there's no third value.
And we know what we wantto do in both cases.
We know that we want toeither ask for a premium
or ask for a discount.
So you can just templatespecialize this completely away
to the point where you havejust completely deterministic,
nonbranching code.
Don't go crazy with this.
Don't remove everysingle if from your code,
it will look really bad,and it'll probably end up
slowing your code down.
But in the hotpath you can go a long way
with absolutely removingbranches completely
from your code.
Hi.
[Audience Member] (distant voice)
The question was, can't theoptimizer optimize this out?
In theory, I guess it probably can.
In practice, it normally doesn't.
And actually there isa very very good point.
So much of this is justguaranteeing that the compiler
is going to do what you expect it to do.
You're kind of lockingthe compiler in to say
hey I really want you toknow that there's only
two particular branches hereor two particular conditions.
So a lot of it is just basicallytaking the risk out of it
and being very very concretewith telling the compiler
what you want to do.
Multi-threading.
I don't like multi-threading.
I know that you need it sometimes.
I know that you don't wantto be doing everything
in the hotpath.
I know that you need to sometimes
maybe have a secondthread on a different CPU
or a different process ona second CPU, second core,
doing some heavy lifting,doing some calculations,
responding to messagesand things like that.
Okay I get that, you need multi-threading,
but keep the data that youshare between the hotpath
and everything else tothe absolute minimum.
If you can do that, you'regonna get away with faster code
than otherwise.
So consider not even sharing data.
Consider just throwing copies of data
across from the producer to the consumer.
May be a lot for a singlewriter single reader queue.
Yeah you're still sharinga little bit there,
you're sharing the atomicread and write pointers
for the queue, but that's actually
not much data to share.
I think that's a reasonable approach.
Of course sometimes youdon't even need to worry
about multithreading, sometimes ...
Sorry, protection over multithreading.
Sometimes receiving outof order is actually okay,
it's not the end of the world.
Sometimes the machinearchitecture is so strong
that it'll actuallyavoid the risk conditions
that you've read about anyway.
So it's worth experimenting with that.
So we're about halfway through now I think
of the tips and tricks section.
The next one is data lookups.
So if you read a softwareengineering textbook,
it'll probably suggest thatif you have an instrument
that you want to trade on, and a market,
and each instrument has one market,
and a market can have many instruments.
So you can see the ER diagram forming.
You'll do a lookup, so you go okay,
this code here would make sense,
so you create your messagethat you want to send
to the exchange and you'llgo look up the market
and you'll get theinformation out of the market
and that will be part of your computation.
So that's fine.
That's not a problem.
But you can do thingsa lot faster than that.
If you're going to read the instrument,
if you're going to read thecurrent price of the instrument,
how many bytes have you read there?
How big's the float?
Four bytes?
Okay, so how many bytes have we read?
64. Why have we read 64?
Because we've read a cacheline.
We can't do anything about that,
that's how the machine works.
So, why not put theinformation that you need
into that cacheline?
So if you know thatyou always need to read
say for example the quantitymultiplier from the market
and that hardly ever changes,well may not change at all
that day, put it into the instrument.
You're going to be readingthat cacheline anyway,
and then you get that for free.
No further lookups, you're done.
It's denormalized, might violate a few
software engineeringprinciples, but that's okay.
You're gonna get faster code.
Containers.
Containers are worth a handful of slides,
so I've got about fourslides on unordered map.
So you need something to do lookups on.
Sometimes you are gonnahave to look up data.
So that's fine.
So this is your likelyimplementation of an unordered map.
You have multiple buckets, you have a key,
the key gets hashed, thehash wall corresponds to one
and only one bucket, andyour key and value pair
will go into the bucket.
And if you get multiplepairs of keys and values
that map to the same bucket,then you'll get a chain effect.
So this will be one linklist underneath the hood.
Default max load factor of one.
So on average, there willbe one item per bucket.
Order one insert, order one find.
Unordered map is kind of your go to map
if you want fast containers within C++
and you want to just use CSDL.
And by the way, the original proposal
was in the bottom right,really worth the read.
It's a really good descriptionof how unordered maps
and unordered sets are implemented.
Link lists are not guaranteedto be in contiguous memory.
So you really don't wantto be jumping around
to this link list looking for your key.
You want to find it the first time.
So, I did an experiment recently
where I took 10,000 ...
Well I generated 10,000keys and I generated them
from the range of zerothrough to one trillion.
Then I inserted all ofthem into a unordered map.
Then I asked H-node how manyother nodes are in your bucket?
And this is the distributionthat I got back.
Most nodes are in abucket all by themselves,
but only just.
You will get collisions.
This was a uniform distribution,
you are going to get collisions.
Unless you have aperfect hashing function.
Which in this case Ideliberately didn't have,
but I had a pretty good hashing function.
Used standard uniform int distribution.
So if we go back, youwill from time to time
have to run through yournodes to find the key.
It's not always going to be the first one.
So let's time this,this is Google Benchmark
for the top half of the slide.
Anyone use Google Benchmark?
Yeah it's good. Good to see.
If you haven't already, check it out
it's fantastic forparamaterizing micro benchmarks.
14 nanoseconds through to 24 nanoseconds,
so you can see that's reasonable.
And you can see thatspeed of a find is broadly
related to the number ofelements in the collection.
There's the Linux Perfoutput in the second half
of the slide there.
Nothing too surprising in there.
Instructions per cyclewas a little bit low,
but there's also quitea few bit of stalling
going on there as well.
And if you do the math,that means that broadly
one out of 500 cache references
is actually a cache miss.
So so.
Or you could use somethinglike Google's dense hash map.
Fast.
A little bit of complexityaround management of collisions,
but that's okay.
So, given the choice I'd go for this.
Or you can ignore Bjarne'ssuggestion this morning
if not reimplementinganything in ECR relation
to hash tables and take a hybrid,
take the best of both worlds.
That's what we do at Optiver,
for a lot of our(mumbles) instead of code,
is we have a slightlydifferent hash table.
Lots of differentimplementations of hash tables,
plenty on the internet,pick your best one.
What I'm going to do isjust quickly describe
what we use at Optiver, 'cuzit's quite a neat approach.
Maybe you do somethingsimilar at your company,
I don't know.
But we have something like this.
So the keys and values are just anywhere,
they could be in a separate collection,
or it could just be heap allocated.
It really doesn't matter too much.
The table itself isactually a precomputed hash
and a pointer to the object.
That's your hash table.
So it's like a hash table of metadata.
Importantly, the hash, eight bytes.
The pointer, eight bytes.
So each pair is 16 bytes.
You can fit four of those in a cacheline.
Which means that when you read one,
you've read another threemore than likely as well.
Which means that you'vealready pre fetched in
any subsequent data that you might need
to resolve the conflict.
So we take an example,if we look for key 73,
if this is an integer key,then well the hash is just 73
as well, maybe this index is to index one.
Go to index one, the hash is 73.
Okay great, we probablyfound what we're looking for,
follow the pointer, yeahthe key is 73 as well.
Okay, great, we're done.
If we take another example,key 12, maps to hash 12,
maybe this thing happensto index slot number three,
based on the implementation.
Lets check the hash, the has is 98,
ah okay, so there's a collision.
That's alright, we'll justfollow it to the right,
what's the next hash value?
Okay, that's hash value12, great that's what we're
looking for, follow thatpointer, yeah, there's our key,
there's our value.
Very cache efficient.
Don't take my word for it.
So if we micro benchmarkthis, twice as fast,
seems to be less susceptibleto the size of the collection.
Instructions per cycle arelooking a lot healthier now,
double if not triple.
And only that one out of1,500 cache references
is actually a cache miss.
So just a more cachefriendly implementation
of a hash table.
There may be a talk later todayabout hash tables like this
possibly by ...
[Audience Member] (indistinct)
Sorry?
[Audience Member] Wednesday.
It's Wednesday.
[Audience Member] Yeah.
Cheers.
There may be a talk onWednesday about this.
Okay a couple more to go.
Always inline and no inline.
The inline keyword isincredibly confusing.
Inline does not meanplease inline, inline means
there may be multipledefinitions, that's okay,
link it please, overlook it.
If you want something to be inlined,
you're far better to useGCC and Clang's attributes
to say force this to be inlined please.
Or force this to not be inlined.
Now, dragons lie here.
You need to be careful about this.
Inlining can make your code faster,
inlining can make your code slower.
Not inlining can also give you
either of those two alternatives.
Again, you need to measure.
Look at the disassembly, buteven better, actually measure
end production.
So just a trivial example here,
but we go to check the market
and if we decide that we'renot going to send the order
'cuz something's gonewrong, do some logging,
otherwise send the order.
Now if this is inlined, ifthe compiler goes hey that's
not a bad function to inline,
it doesn't take too many parameters,
yeah sure lets inline it, why not?
You're gonna polluteyour instruction cache.
You've got all of this logging code,
which you don't reallywant in the hotpath,
right there in the middle of the hotpath.
So this is an easy one,just no inline this,
that'll drop out, that'll turn to a call,
chances are the branch predictor will ...
Well two things, chancesare the compiler will
pick the right branch, and even better
at runtime the branchpredictor will probably
be picking the right branchall the time for you.
Gonna talk about the branch predictor more
very soon by the way.
The hardware branch predictor.
I'm gonna talk about the hardwarebranch predictor right now
as it turns out.
Okay, this is the most important slide
out of the 15 or so tips and tricks.
This gives me a fivemicrosecond speed up on my code.
Or in other words,whenever I screw this up,
the systems run five microseconds slower.
So much in fact, I havea flow chart on my desk
which basically goes ohyou have five markers
you're going slower.
Yes. Did you screw up?
Dump your orders. Yes.
Fix it, and go back to the start.
So this is what happens in practice.
You hardly ever executethe fastpath, the hotpath,
and I said this right at the start.
Gonna say it again, the fastpathis very seldomly executed.
Maybe you get close and thenyou decide actually I can't
because of risk limits,the next market data
that comes in isn't interesting,
the next is, but againthe strategy decides
I've already got enoughof that instrument,
I don't want to trade it.
Then eventually, at some pointah great there's an order
we want to go for, let's shoot for it.
Now on top of this there'sother things going on
which I haven't put onthe board, on the graph,
things like the unorderedmap that's trampling
all over cache and otherthings that are trampling
all over our cache,handling other messages
that are coming in, doingbackground processing.
So how do we fix this?
How do we keep the cache hot?
Well, we pretend we livein a different universe
where everything that wedo results in an order
being sent to the exchange.
Here's a tip, you reallydon't want to send
everything to the exchange.
They'd get very annoyedwith you very quickly.
But you can pretend.
So as long as you've got confidence
that you can stop this beforeit gets to the exchange
within your own software,within your own control,
then pick a number somewherebetween 1,000 to 10,000.
That's gonna be the number of times
that you simulate sending anorder through your system.
If you're using a low latency network card
such as Mellanox or Solar Flare
chances are even the cardwill allow you to do this.
This is industry practice,
it understands that peoplewant to push data onto the card
but not send it.
It's just warming the card.
So network cards will supportthis, so that's great.
So basically saturate your system with
I'm going to send an order,I'm going to send an order,
I'm going to send an order,bang, I did send an order.
Keeps your instruction cache hot, yeah?
Nothing's gonna get evictedif you're doing that.
Probably keeps your data cache hot
if you're picking the right data,
and it trains yourhardware branch predictor.
Your branch predictor, ifit's just done the same thing
10,000 times before in the last second,
chances are it's gonna pickthe right path this time.
Hey.
[Audience Member] Do you useprofile-guided optimization?
Ah yeah, good question.
Do I use profile-guided optimization?
Yeah, sometimes, in many different ways.
Yeah, so profile-guided optimization
is kind of a substitute for this,
but again that's only somethingyou can do at compile time
and it's not gonna helpwith the hardware runtime.
But indeed, yeah profile-guidedoptimization is great
except you can overfit the model.
So profile-guidedoptimization is effectively,
you run a simulation, ormaybe even run a production,
where you're writing to a profiling file
of what's actuallyhappening with your system,
then you recompile with that information
and then the compiler can usethat for branch prediction
hints and reordering.
So that works relatively well,
but you need to makesure that your profiling
that you're doing matches production.
And sometimes that means thatyou're very fast sometimes
and way off the mark other times.
So it's kind of this thingthat looks really great
and it is some of the time.
Good question.
Okay, this is a Xeon E5 CPU.
How many cores does this have?
I claim eight.
And the cores are onthe left and the right.
And what's in the middle?
Cache. Glorious, glorious cache.
L3 cache. Probably about50 megs worth of cache.
Very fast, this is what you want.
How many cores get access to this cache?
All of them.
Well that sounds like a problem.
Because I want this cache all to myself.
Yeah?
I turn off all but one of your cores.
Now you get the cache all to yourself.
It's a neat trick, it'snot very respectful
to any intel engineers in the room today
and as a teenager Icouldn't believe I'd be
having eight cores andworking for a company
where I could turn allbut one of them off.
Or 22 cores and turningall of one of them off.
It's a really effective wayof increasing the average
L3 cache per core.
(audience laughs)
Hey.
Sorry?
Have I tried using a single core CPU?
Oh (laughs), interesting.
So have I tried using a single core CPU
or convincing someone to make one?
So no I haven't, for the answerto both of your questions.
We can talk about thatafterwards a little bit,
but I suspect that Intel arenot necessarily going to be
particularly interestedin what I would want,
because my domain is such asmall part of Intel's market.
That's just my gut feel.
Oh incidentally, if I can go back one.
Yeah if you do run multiple cores,
normally you won't disable your cores,
but just be careful about whatother processors are running
on other cores that share that cache.
If they're noisy it's gonna cost you.
If you've got noisyneighbors, get rid of them,
put them on a completelydifferent CPU somewhere else.
It will probably help you out.
Great quote.
(audience laughs)
Okay this is nitpicking,but this caught me out
and it caught people fromother trading companies
out as well.
Placement new can be slightlyinefficient. Slightly.
So if you use any of these compilers,
placement new will bedoing a null pointer check.
And if the pointer that youpass into placement new is null
it won't construct the object,it won't destruct the object,
and it will return nullback to the caller.
Why?
Because that's what regular new does.
And the spec was a little bit ambiguous
as to what placement new was meant to do.
So most compiler implementersjust did the same thing
as regular new.
But it's actually really inefficient.
Why would C++ do anadditional check for you?
That's not the idea of C++,
you don't pay for what you don't get.
You don't pay for what you don't use.
If you're going to passa pointer into a function
you should make sure thatit's null or not null.
Yeah, so.
Marc Glisse and JonathanWakely got the spec updated,
the standard updated, sothat's now undefined behavior
to pass null on to placement new.
That means that the compilercan stop doing this null
pointer check.
Does a single null pointercheck make a difference?
Actually it does, yeah.
So if you were right onthe border of this function
being inlined and not beinginlined and this additional
instruction could push it over the top.
Also with GCC, GCC wasparticularly sensitive to this,
and actually gave up on awhole bunch of optimizations.
So yeah, just a smallthing, but it actually made
quite a big difference,enough that several
trading companies picked up on this.
If you can't use a modernversion of the compiler
that has the effects,there's a quick workaround
that will work around it, at least in GCC.
Okay another thing that caught me out is
GCC five and below or belowGCC 5.1, standard string
it wasn't a great implementation.
Had copy on right semanticsand null small string
optimization that wasn't there.
So great, we moved tolater versions of GCC,
started using strings all over the place,
but what do you know, it was slow.
Because we use a standarddistribution of Redhat and Centos
and the way that GCC ispackaged is it maintains
ABI backwards compatibility.
So it doesn't do the standardstring breaking change,
which means that we'restill stuck with the old
standard string implementation.
And that was first noticedas being a bit questionable
by Herb Sutter in 1999, and today in 2017
I'm still stuck withcopy-on-write strings.
Another small nitpick, but C++11 has this
static local variable initialization
guarantees that staticwill be initialized once
and only once even if multithreaded.
Even if an exception comes inand blows away that routine
it'll go okay we're not initialized yet.
So very useful, but of course,that doesn't come for free.
So every time that you referto that static variable
inside that function there'sgoing to be a slight cost,
which is checking the guard variable
to see if that has been set or not.
So again it's a minor nitpick,
but if you are absolutelymicro optimizing code,
this will have a slight impact as well.
Who knows that standardfunction might allocate?
Most people? Yep, cool.
So that can be a little bit surprising,
because sometimes it doesn't allocate.
It depends on the implementation.
You can do a small functionoptimization in there
around about 16 bytes I think.
So this is with GCC 7.2, thisis a silly trivial example,
but here we are, we're notactually doing anything,
we're capturing 24 bytes worth of data
but we're not doing anything with it.
Clang will just optimizethis right out, GCC doesn't.
So you'll actually seethat there's an allocation
for this code and actually deallocation,
which I didn't print on the screen.
That gets a little bit annoying,
because standard functionsare actually very very useful.
So you'll see that severalpeople have implemented
nonallocating versions.
I believe there's talks on this,
again I would possiblyclaim later on today,
but maybe I'm wrong again.
But yeah if you want to,
this is a shameless plug by the way,
but go check out SG14's inplace function,
which doesn't allocate.
It will allocate inplace,which means that if you declare
this function on the stack,the buffer for the closure
will be on the stack withcompile time static asserts.
It also runtime checks if need be as well.
Okay, the very lastpoint for this section,
then we've only got a small section to go,
is glibc can be particularly evil,
if you're looking to do low latency stuff.
So standard power, if youcall it with one to the power
of 1.4, you'll get the right answer.
If you call it 1.0 to the power of 1.5,
you're also gonna get the right answer,
but one's gonna be wayslower than the other.
It's not quite 1.0 by the way,it's 1.0 with a little bit
of rounding error in there.
But if you look at these timings,
if we go on to 1.4, 53nanos on both an old version
of glibc and a newversion of glibc, great.
If you do it with 1.5the pathological case,
you're in trouble.
Half a millisecond to calculate one single
computation of power.
The reason is thefunction's transcendental,
it'll try to be accurate fast,
if not it'll try to be accurate slow.
So this has caught us out,it's effectively crashed
our autotraders, because ifyou try to calculate this
a thousand times quickly,everything kind of stops.
Yeah, relatively surprising,you can upgrade to glibc 2.21
which is a couple ofyears old now I think,
that'll help you out, butmost standard distros of Linux
will be packaged with 2.17.
So again, something to watch out for.
While I'm on this topic, another thing
which isn't on my slidesbut I should mention,
is try to avoid system calls.
System calls will just kill you.
So you don't want tobe going to the kernel,
at all, you want justyour C++ code to be run.
You wanna get allinterrupts absolutely away
from the call that you're running.
You want your hotpath to berunning on a single call.
Nothing else needs to knowabout that single call.
Any sort of system calls,like a call to select
or a call to kernal space tcp or anything
that might invoke asystem call whatsoever,
you want to get rid of that.
Sorry, just popped into myhead, I thought I should say it.
Not particularly funny, but very true.
So the final section, onlya couple of slides actually.
Hey! I got a reminder.
Talk, yes, that is right.
(audience chuckles)
That could've gone far worse.
Okay, there's two ways tokind of measure things really,
or two common approaches.
One is to profile, seewhat your code is doing.
The other one's to actuallybenchmark and time it.
These two things are subtly different.
So profiling is good,it can show you hey look
you're spending mostof your time in Malloc.
Why are you even in Malloc?This makes no sense.
Whereas benchmarking isactually start, stop,
right this is how fast we are.
If you make an improvement toyour code based on profiling,
that's great, but maybeyour code's not faster,
maybe it's slower.
You know, you're justguessing. You need to measure.
Once you've finishedmeasuring, measure again,
and then measure onemore time, a third time.
Okay so gprof, a sampling profiler.
Gprof is great for checkingout what your code is doing,
but it's not going towork if most of the time
your code is doing nothing and then
for 300 or 400 nanoseconds,all of a sudden
it's doing something and thenit goes back to being idle.
Gprof is going to tell you
that you're doing absolutely nothing.
It's gonna miss the actualevents that you care about.
Gprofs are out.
Valgrind, callgrind, fantastic.
Can tell you actuallywhat your code is doing,
which functions it's going into,
how much memory you're using.
But it's basically a virtual machine,
it's simulating the CPU, itdoesn't even take into account
I/O characteristics, that's gone as well.
It's too intrusive, it'snot going to give you
accurate performance results.
I've been talking aboutmicrobenchmarking a lot.
It's kind of okay, but don't rely on it.
Because if you're sittingthere spinning in an absolute
tightloop, doing map lookups,that's a very different
characteristic from whatyour code is gonna be doing
in production when it'sdoing a one off map lookup
once every eight trillion cycles.
So benchmarks are kindof nice for map shootouts
a little bit and things like that,
but you can only take them so far.
Now all of these tools are really useful,
don't get me wrong I useall of them, they're great,
but not necessarilyfor that very last step
of microoptimizing your code.
So we need something else.
So this is what I use, and itis by far the most accurate
benchmarking system I can come up with.
What you do is you actuallyset up a production system,
the server on the left hereis replaying market data
in real time.
Across a network, my codeis running on the server
on the right, picking up the market data
and then from time to time sending a trade
or a request to trade back across the wire
to where it thinks is the exchange.
Of course it's just a fake exchange.
What I've got in the middle is a switch,
and the switch is tapped into the network
and it's unobtrusivelycapturing every packet
that's going across thewire and recording it,
and also putting a timestampinto either the header
or the footer of the evernet packet.
So actually recording that true time
that that packet went across the wire.
Probably within an accuracyof five nanoseconds
or something like that.
Then afterwards, once you've done an hour
or a days worth of simulated trading,
then you can analyze the packets,
look at which packets triggered an event,
compare the timestamp ofthe corresponding packet
that got sent back out toexchange, do a difference
of the timestamps and thatwill tell you the actual
true time that it's takingfor your system to run.
And if you see a speedupin this lab setup,
you'll see a speedup in production.
If you see a slowdown in the lab,
you'll see a slowdown in production.
Incredibly hard to set up,you have to get this perfect,
all the time you'll forget ...
You forgot to turn offparticular interwraps
or you forgot to set CPU affinity,
or someone decided it might be a good idea
to build on that server because they found
that it was free and veryfew people were on it
so they ran GCC on 24 cores.
But this is a real pain,it's not easy to set up,
but it's by far the most accurate way
to do true measurement.
Okay, so the brief summary
is you don't need tobe an expert about C++,
but you do need to know it relatively well
if you want fast low latency software.
You have to understand your compiler.
You have to understand whatyour compiler's trying to do
and help your compileralong from time to time.
Knowing the machinearchitecture will help.
Knowing the impact of cache,the impact of other processes,
it's gonna help you makecreate design decisions.
Ultimately, aim for reallyreally simple logic.
Not just simple logic, but getrid of your logic if you can,
go for static asserts, consexpressions, templates,
just try to make your compilerslife as easy as possible
in that respect orreduce your code as much.
The best thing about thisis compilers optimize
simple code the best by far.
Simple code runs faster than complex code.
It's just a fact, just as ...
Sometimes you don'tneed perfect precision,
sometimes an approximationis absolutely fine
and if the approximationis fast, that's great.
There's no point calculatinga theoretical price
to 12 significant digits ifreally only two digits would do.
And if you need to do expensive work,
don't do it on the hotpath,just do it when you think
it's a quiet time.
And ultimately it's about measurement.
If you can't measure, you'rebasically just guessing.
It's incredible, every time that I think
this code is gonna be faster,
do do do do, measureit, it's always slower.
I always get it wrong, it's difficult.
Okay, so that's me.
This was quite lightweight,not entirely scientific,
it's more just a hey thisis the sort of things
that we do.
I hope you found that somewhat useful.
Did people learn something?
Little bit? So-so?
Okay, cool.
I'm pleased. Thank you.
Cheers.(audience applauds)
We've got a couple minutes for questions
if anyone feels like it.
Hey.
[Audience Member] Hi, can you comment
in the use cases when youwere only using one core
on the multicore chip, do youknow if Intel's Turbo Boost
was helping you out at all or not?
Yeah, that's a good question.
So I could spend anotherhour or two talking
about the actual system tuning type of it.
Won't go into too much detail here,
but effectively youneed to be very careful
of Turbo Boost and different states,
you just want things to be constant.
You don't want things tobe jumping up and down
and yeah ...
Cool.
Hey.
Oh, sorry, we'll go there.
Hi.
[Audience Member]Hi, so one quick thing,
the actual question ison newer architectures,
you can actually force theL3 cache to be partitioned?
Yeah.- [Audience Member] Okay.
You already know, but I'mtalking about other people.
I'll just recap incase no one heard that,
or I just cut you off.
Yeah, you don't justhave to turn off CPUs,
you can actually lock thecache with latest versions
of Intel CPUs.
[Audience Member] Right.
When you describe the hash table,
the layout in memory, you had an array,
and you had the full hash, which you keep
and then a pointer, but usuallyin HFT you only really care
about access, not insertion.
Insertion's often outsideof the critical path.
Correct, yeah.- [Audience Member] Not always
but often.
So why not instead have ...
If you have the key valuesin an array as well,
you can just use theindex of the hash itself
to access that.
Now you're only storing thehashes in the first lookup array
so you get eight instead of four.
So why not?
If I understand the question correctly,
won't that put morepressure on your cache?
Because you'd actually haveless hash values per cache line.
[Audience Member] You'llhave more, because you're not
storing the pointer at all.
The first layer of thelookup is just into an array
with nothing but the hashes.
Oh I see what you're saying.
[Audience Member] Andthen if you verified
that the hash is correct,you just use the index
and go into the key value array.
Yeah you could do that,but that's going to also,
as long as you're very confidentthat you're not getting
too many collisions.
So I think you're going to get a tradeoff.
I see what you're saying,and yeah you could do that,
but that's assuming a prettydamn good hash function
I think.
But there's always ...
There's no right and wronganswer I think for hash tables
there's always a bit of a sliding scale.
[Audience Member] Alright.
Thank you, cheers.
Hey.
[Audience Member] SoI'm interested about
where you talked about using only one CPU
unlocking the outlet.
Others are one core.
Do you mean that you're goingto use one thread as well?
Yeah.
[Audience Member] And how does that work
when you have multiple datasources that you need to
look at and everything?
Like ...
Yeah.
[Audience Member] Can youexplain a bit about that?
Yeah sure, so this isgetting away from the C++ side,
which I purposely didn'twant to talk about,
being a C++ conference,but very very briefly,
yeah exactly, one thread on that one core.
That's it, that's the fast thread.
But of course you can from timeto time pull data coming in
once every few hundred milliseconds,
then pull the data, andstatistically you should be okay.
It all very much depends onhow much data's coming in,
if you've got a lot of data coming in,
then no you'd need a second thread.
But yeah, effectively ...
[Audience Member] I meanthough the data that you need
to respond quickly, thedata from the market.
So you can't actually pullthem, because then you reduce
your latency.
Ah yeah, true, but youcould also have something
ahead of you doing a littlebit of a low pass filter
or discarding more of the data.
Yeah, you definitelyhave a component ahead
filtering a lot of that out for you,
otherwise yeah you'd be saturated.
[Audience Member] I see.- Yeah.
[Audience Member] Thank you.
Yeah, you're welcome.
Hi.
[Audience Member] Just a small point.
You said consider deletinglarge objects in another thread,
one of the things I've been burned by
is that we'll drain the TCMalluc size class for that
and then cause global Mutexlocks across all your threads
when you allocate ordeallocate in that size class.
Yeah, I was wondering if someone
would pick up on that (laughs).
True.
We actually often will allocateon that separate thread
as well and just pass the pointer across.
Yeah, good point.
I was praying no onewould pick up on that.
Hi.
[Audience Member] Hi,do you use things like
built in expect to markbranches slightly or unlikely?
Yeah, so this will have tobe the last question correct?
Yeah, and I deleted thosefrom my slides this morning
because I ran out of time,as we have just done as well.
I mean a built in expect,definitely for those who
don't know about it,it's just a macro that's
will tell the compilerhey this is the branch
that I want you to take.
Absolutely, you can use that,but it's not a silver bullet.
Everything is about gettingthat hardware branch predictor
trained up.
In saying that, if you have a function
which is called very very irregularly
and you want that functionto be fast when it's called.
The hardware branch predictorhas zero information.
In that case, yeah, yourlikely true and likely false
branch prediction hitsare exactly what you need
to make sure that correct path is hit.
Good question, thank you.
So we've gotta ...
We can catch up afterwards,if maybe we have to.
Times up now. We can talk afterwards.
Yeah.
Great, thank you.
(audience applauds)