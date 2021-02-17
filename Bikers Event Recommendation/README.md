# Solution for Pattern Recognition and Machine Learning Data Contest on Kaggle

https://www.kaggle.com/c/prml-data-contest-nov-2020/overview

Welcome to the PRML Nov-2020 data contest.
Here you will build prediction models for members of Biker-Interest-Group and try to predict which bike-tours will be of interest to bikers. You can use all the extra information provided, such as biker's previous interests, his/her demographic details, past tours, friend circle etc. This dataset is derived from a real world scenario; so take care of sanitizing/handling real data.

To learn about the bikers, you will have access to their feature information and friends-network. Information regarding tours and its participants is also available. Use the data in creative ways to come up a ML model that predicts bikers preference among tours. Supervised data is available about biker's preference is available in train.csv.

Please note that some entries in the real-data might be missing due to bikers incomplete profile; students are advised to take care of such missing entries.
The following 6 files are available for the data contest.

**train.csv**
train.csv has rows corresponding to tours shown to a biker, and data about whether he/she liked the tour or not.
- biker_id: Unique identifier for a biker.
- tour_id: Unique identifier for particular tour.
- invited: {0/1} bool variable to denote if the biker was invited to the particular tour.
- timestamp: Approximate time when the biker was informed about the tour.
- like: {0/1} bool variable as per the entry made by biker. 1 indicates biker has liked the tour. 0 indicates that he has not responded to the 'like' question.
- dislike: {0/1} bool variable as per the entry made by biker. 1 indicates biker has not-liked the tour. 0 indicates that he has not responded to the 'not_like' question.
NOTE: It is possible that the biker simply ignored the questions and did not respond to both 'like' and 'dislike' entries, hence values maybe 0,0 for both last columns.

**test.csv**
test.csv same as above train data, except for the last two columns of like/dislike
- biker_id: Unique identifier for a biker.
- tour_id: Unique identifier for particular tour.
- invited: {0/1} bool variable to denote if the biker was invited to the particular tour.
- timestamp: Approximate time when the biker was informed about the tour.

**tour_convoy.csv**
tour_convoy.csv consists the list of bikers that showed interest in a particular tour.
- tour_id: Unique identifier for particular tour.
- going: Space-delimited list of bikers who said they will go to the tour.
- maybe: Space-delimited list of bikers who said they might go to the tour.
- invited: Space-delimited list of bikers who were invited to the tour.
- not_going: Space-delimited list of bikers who said they will not go to the tour.

**bikers.csv**
bikers.csv has feature information about bikers.
- biker_id: Unique identifier for a biker person.
- language_id: Identifier of the language biker speaks.
- location_id: Identifier of the location biker resides in.
- bornIn: Year of birth of the biker to estimate their age.
- gender: male/female based on their bikers input.
- member_since: Date of joining the bikers interest group.
- area: bikers location (if known).
- time_zone: this is the offset in minutes to GMT timezone. (For example Indian Time is +5:30 GMT, so +330 minutes).

**tours.csv**
tours.csv consists feature information about the tours.
- tour_id: Unique identifier for particular tour.
- biker_id: ID of the biker who organized the tour.
- tour_date: date on which tour was conducted.
- city: location of tour (if known)
- state: location of tour (if known)
- pincode: location of tour (if known)
- country: location of tour (if known)
- latitude: approximate location of the starting point of the tour (if known)
- longitude: approximate location of the starting point of the tour (if known)
- w1, w2, ..., w100: Number of occurences of most common words in the description of the tour. We took 100 most common/important words among all the descriptions provided in the tour guide, and each column w1, w2, ... w100 gives the count of number of times each word w_i has occured in the description of a given tour_id.
- w_other: count of other words.

**bikers_network.csv**
bikers_network.csv consists of the social networks of the bikers. This is derived from the group of bikers that are know each other via some groups.
- biker_id: unique id for a biker
- friends: this is a list of all friends of given biker_id (Note: this is a space delimited column).

**PLEASE NOTE**
We don't have data of invitees/likes of all tours in tours.csv and tour_convoy.csv, and hence any given biker/tour in tours.csv or tour_convoy.csv may not appear in train/test.csv.
