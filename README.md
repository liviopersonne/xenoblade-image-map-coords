This is an unfinished attempt I made from a long time ago (Feb 2024).

I tried to use image of maps from Xenoblade paired with data from the website [ookaze](https://www.ookaze.fr/Xenoblade/), to draw spawn points of objects on the map.

To do this, I analysed the image and tried to detect the grid that was shown on it using `pytesseract`, and then calculate the transformation that switched between the map coordinates and the image coordinates.

At the time of writing this readme, I have not looked at this code for 1.5 years, but I remember abandonning the project because I couldn't get it to work correctly. But the idea is interesting so that's why I'm still publishing this.
