# 🐾 Animal Classification Dataset

## Overview
- **Total Classes:** 94
- **Total Images:** 139,208
- **Train Set:** 108,016 images
- **Validation Set:** 15,456 images
- **Test Set:** 15,736 images

> **Note:** The deployed inference model (model/animal_classification_mobilenetv3_large.pth) classifies across **93 species** (all dataset species except newly added goose).

## Dataset Split Breakdown

| Species | Train Set | Validation Set | Test Set | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Alpaca** | 95 | 21 | 22 | 138 |
| **Angora** | 866 | 174 | 206 | 1,246 |
| **Antelope** | 296 | 30 | 30 | 356 |
| **Armadillo** | 890 | 185 | 176 | 1,251 |
| **Badger** | 1,719 | 221 | 188 | 2,128 |
| **Barn Spider** | 922 | 171 | 207 | 1,300 |
| **Bat** | 700 | 55 | 59 | 814 |
| **Bear** | 752 | 109 | 110 | 971 |
| **Bee** | 5,181 | 976 | 956 | 7,113 |
| **Beetle** | 408 | 7 | 8 | 423 |
| **Bison** | 1,364 | 200 | 211 | 1,775 |
| **Boar** | 411 | 26 | 27 | 464 |
| **Bullfrog** | 914 | 207 | 179 | 1,300 |
| **Camel** | 368 | 20 | 20 | 408 |
| **Capybara** | 278 | 20 | 20 | 318 |
| **Cat** | 4,356 | 500 | 500 | 5,356 |
| **Caterpillar** | 813 | 72 | 79 | 964 |
| **Cheetah** | 2,267 | 206 | 208 | 2,681 |
| **Chimpanzee** | 1,381 | 194 | 187 | 1,762 |
| **Cock** | 900 | 210 | 188 | 1,298 |
| **Cockroach** | 1,378 | 188 | 221 | 1,787 |
| **Colobus** | 923 | 182 | 194 | 1,299 |
| **Cougar** | 957 | 175 | 191 | 1,323 |
| **Cow** | 728 | 106 | 115 | 949 |
| **Coyote** | 1,466 | 197 | 218 | 1,881 |
| **Crocodile** | 1,011 | 130 | 140 | 1,281 |
| **Crow** | 550 | 28 | 25 | 603 |
| **Deer** | 732 | 91 | 96 | 919 |
| **Dog** | 4,375 | 500 | 500 | 5,375 |
| **Dolphin** | 892 | 111 | 86 | 1,089 |
| **Donkey** | 445 | 5 | 9 | 459 |
| **Dragonfly** | 1,420 | 226 | 211 | 1,857 |
| **Duck** | 1,113 | 174 | 166 | 1,453 |
| **Eagle** | 924 | 114 | 122 | 1,160 |
| **Elephant** | 1,124 | 145 | 170 | 1,439 |
| **Flamingo** | 1,890 | 277 | 298 | 2,465 |
| **Fly** | 1,284 | 206 | 219 | 1,709 |
| **Fox** | 575 | 33 | 41 | 649 |
| **Gazelle** | 877 | 201 | 221 | 1,299 |
| **Giraffe** | 1,284 | 114 | 107 | 1,505 |
| **Goat** | 617 | 45 | 43 | 705 |
| **Goldfish** | 1,427 | 221 | 214 | 1,862 |
| **Goose** | 1,448 | 226 | 243 | 1,917 |
| **Gorilla** | 1,332 | 225 | 208 | 1,765 |
| **Grasshopper** | 1,320 | 186 | 222 | 1,728 |
| **Guinea Pig** | 905 | 200 | 192 | 1,297 |
| **Hammerhead** | 912 | 197 | 191 | 1,300 |
| **Hamster** | 1,368 | 223 | 213 | 1,804 |
| **Hare** | 1,328 | 191 | 208 | 1,727 |
| **Hippopotamus** | 1,403 | 213 | 223 | 1,839 |
| **Hog** | 922 | 181 | 197 | 1,300 |
| **Hyena** | 1,334 | 177 | 229 | 1,740 |
| **Indri** | 942 | 182 | 174 | 1,298 |
| **Jaguar** | 1,151 | 16 | 20 | 1,187 |
| **Jellyfish** | 1,715 | 267 | 290 | 2,272 |
| **Kangaroo** | 576 | 126 | 150 | 852 |
| **Koala** | 3,857 | 177 | 238 | 4,272 |
| **Ladybugs** | 439 | 5 | 9 | 453 |
| **Langur** | 924 | 186 | 188 | 1,298 |
| **Leatherback Turtle** | 907 | 197 | 189 | 1,293 |
| **Leopard** | 1,713 | 188 | 208 | 2,109 |
| **Lion** | 1,917 | 192 | 219 | 2,328 |
| **Llama** | 950 | 215 | 208 | 1,373 |
| **Marmoset** | 925 | 191 | 184 | 1,300 |
| **Mink** | 930 | 186 | 179 | 1,295 |
| **Mongoose** | 935 | 181 | 176 | 1,292 |
| **Mosquito** | 889 | 75 | 83 | 1,047 |
| **Mule** | 58 | 13 | 13 | 84 |
| **Okapi** | 60 | 13 | 19 | 92 |
| **Orangutan** | 1,803 | 192 | 206 | 2,201 |
| **Porcupine** | 936 | 225 | 194 | 1,355 |
| **Possum** | 439 | 7 | 9 | 455 |
| **Proboscis Monkey** | 909 | 179 | 208 | 1,296 |
| **Puffer** | 915 | 212 | 173 | 1,300 |
| **Red Panda** | 48 | 8 | 16 | 72 |
| **Reindeer** | 39 | 11 | 8 | 58 |
| **Rock Crab** | 903 | 182 | 194 | 1,279 |
| **Scorpion** | 1,698 | 304 | 255 | 2,257 |
| **Seahorse** | 428 | 34 | 27 | 489 |
| **Seal** | 43 | 9 | 6 | 58 |
| **Snail** | 1,534 | 317 | 329 | 2,180 |
| **Snake** | 1,315 | 199 | 184 | 1,698 |
| **Snow Leopard** | 1,282 | 180 | 202 | 1,664 |
| **Spider Monkey** | 895 | 197 | 201 | 1,293 |
| **Squirrel** | 1,204 | 163 | 172 | 1,539 |
| **Tarantula** | 962 | 179 | 159 | 1,300 |
| **Terrapin** | 916 | 198 | 186 | 1,300 |
| **Tiger** | 2,894 | 179 | 202 | 3,275 |
| **Tortoise** | 196 | 40 | 24 | 260 |
| **Warthog** | 925 | 195 | 177 | 1,297 |
| **Weasel** | 1,121 | 172 | 188 | 1,481 |
| **Wolf Spider** | 879 | 192 | 225 | 1,296 |
| **Wombat** | 1,685 | 217 | 184 | 2,086 |
| **Zebra** | 2,014 | 163 | 146 | 2,323 |
| **Total** | **108,016** | **15,456** | **15,736** | **139,208** |

## Per-Class Breakdown:
- **Alpaca:** test-22, train-95, val-21 (total: 138)
- **Angora:** test-206, train-866, val-174 (total: 1,246)
- **Antelope:** test-30, train-296, val-30 (total: 356)
- **Armadillo:** test-176, train-890, val-185 (total: 1,251)
- **Badger:** test-188, train-1,719, val-221 (total: 2,128)
- **Barn Spider:** test-207, train-922, val-171 (total: 1,300)
- **Bat:** test-59, train-700, val-55 (total: 814)
- **Bear:** test-110, train-752, val-109 (total: 971)
- **Bee:** test-956, train-5,181, val-976 (total: 7,113)
- **Beetle:** test-8, train-408, val-7 (total: 423)
- **Bison:** test-211, train-1,364, val-200 (total: 1,775)
- **Boar:** test-27, train-411, val-26 (total: 464)
- **Bullfrog:** test-179, train-914, val-207 (total: 1,300)
- **Camel:** test-20, train-368, val-20 (total: 408)
- **Capybara:** test-20, train-278, val-20 (total: 318)
- **Cat:** test-500, train-4,356, val-500 (total: 5,356)
- **Caterpillar:** test-79, train-813, val-72 (total: 964)
- **Cheetah:** test-208, train-2,267, val-206 (total: 2,681)
- **Chimpanzee:** test-187, train-1,381, val-194 (total: 1,762)
- **Cock:** test-188, train-900, val-210 (total: 1,298)
- **Cockroach:** test-221, train-1,378, val-188 (total: 1,787)
- **Colobus:** test-194, train-923, val-182 (total: 1,299)
- **Cougar:** test-191, train-957, val-175 (total: 1,323)
- **Cow:** test-115, train-728, val-106 (total: 949)
- **Coyote:** test-218, train-1,466, val-197 (total: 1,881)
- **Crocodile:** test-140, train-1,011, val-130 (total: 1,281)
- **Crow:** test-25, train-550, val-28 (total: 603)
- **Deer:** test-96, train-732, val-91 (total: 919)
- **Dog:** test-500, train-4,375, val-500 (total: 5,375)
- **Dolphin:** test-86, train-892, val-111 (total: 1,089)
- **Donkey:** test-9, train-445, val-5 (total: 459)
- **Dragonfly:** test-211, train-1,420, val-226 (total: 1,857)
- **Duck:** test-166, train-1,113, val-174 (total: 1,453)
- **Eagle:** test-122, train-924, val-114 (total: 1,160)
- **Elephant:** test-170, train-1,124, val-145 (total: 1,439)
- **Flamingo:** test-298, train-1,890, val-277 (total: 2,465)
- **Fly:** test-219, train-1,284, val-206 (total: 1,709)
- **Fox:** test-41, train-575, val-33 (total: 649)
- **Gazelle:** test-221, train-877, val-201 (total: 1,299)
- **Giraffe:** test-107, train-1,284, val-114 (total: 1,505)
- **Goat:** test-43, train-617, val-45 (total: 705)
- **Goldfish:** test-214, train-1,427, val-221 (total: 1,862)
- **Goose:** test-243, train-1,448, val-226 (total: 1,917)
- **Gorilla:** test-208, train-1,332, val-225 (total: 1,765)
- **Grasshopper:** test-222, train-1,320, val-186 (total: 1,728)
- **Guinea Pig:** test-192, train-905, val-200 (total: 1,297)
- **Hammerhead:** test-191, train-912, val-197 (total: 1,300)
- **Hamster:** test-213, train-1,368, val-223 (total: 1,804)
- **Hare:** test-208, train-1,328, val-191 (total: 1,727)
- **Hippopotamus:** test-223, train-1,403, val-213 (total: 1,839)
- **Hog:** test-197, train-922, val-181 (total: 1,300)
- **Hyena:** test-229, train-1,334, val-177 (total: 1,740)
- **Indri:** test-174, train-942, val-182 (total: 1,298)
- **Jaguar:** test-20, train-1,151, val-16 (total: 1,187)
- **Jellyfish:** test-290, train-1,715, val-267 (total: 2,272)
- **Kangaroo:** test-150, train-576, val-126 (total: 852)
- **Koala:** test-238, train-3,857, val-177 (total: 4,272)
- **Ladybugs:** test-9, train-439, val-5 (total: 453)
- **Langur:** test-188, train-924, val-186 (total: 1,298)
- **Leatherback Turtle:** test-189, train-907, val-197 (total: 1,293)
- **Leopard:** test-208, train-1,713, val-188 (total: 2,109)
- **Lion:** test-219, train-1,917, val-192 (total: 2,328)
- **Llama:** test-208, train-950, val-215 (total: 1,373)
- **Marmoset:** test-184, train-925, val-191 (total: 1,300)
- **Mink:** test-179, train-930, val-186 (total: 1,295)
- **Mongoose:** test-176, train-935, val-181 (total: 1,292)
- **Mosquito:** test-83, train-889, val-75 (total: 1,047)
- **Mule:** test-13, train-58, val-13 (total: 84)
- **Okapi:** test-19, train-60, val-13 (total: 92)
- **Orangutan:** test-206, train-1,803, val-192 (total: 2,201)
- **Porcupine:** test-194, train-936, val-225 (total: 1,355)
- **Possum:** test-9, train-439, val-7 (total: 455)
- **Proboscis Monkey:** test-208, train-909, val-179 (total: 1,296)
- **Puffer:** test-173, train-915, val-212 (total: 1,300)
- **Red Panda:** test-16, train-48, val-8 (total: 72)
- **Reindeer:** test-8, train-39, val-11 (total: 58)
- **Rock Crab:** test-194, train-903, val-182 (total: 1,279)
- **Scorpion:** test-255, train-1,698, val-304 (total: 2,257)
- **Seahorse:** test-27, train-428, val-34 (total: 489)
- **Seal:** test-6, train-43, val-9 (total: 58)
- **Snail:** test-329, train-1,534, val-317 (total: 2,180)
- **Snake:** test-184, train-1,315, val-199 (total: 1,698)
- **Snow Leopard:** test-202, train-1,282, val-180 (total: 1,664)
- **Spider Monkey:** test-201, train-895, val-197 (total: 1,293)
- **Squirrel:** test-172, train-1,204, val-163 (total: 1,539)
- **Tarantula:** test-159, train-962, val-179 (total: 1,300)
- **Terrapin:** test-186, train-916, val-198 (total: 1,300)
- **Tiger:** test-202, train-2,894, val-179 (total: 3,275)
- **Tortoise:** test-24, train-196, val-40 (total: 260)
- **Warthog:** test-177, train-925, val-195 (total: 1,297)
- **Weasel:** test-188, train-1,121, val-172 (total: 1,481)
- **Wolf Spider:** test-225, train-879, val-192 (total: 1,296)
- **Wombat:** test-184, train-1,685, val-217 (total: 2,086)
- **Zebra:** test-146, train-2,014, val-163 (total: 2,323)

