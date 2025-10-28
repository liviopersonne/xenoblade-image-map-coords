from PIL import Image
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Site: https://www.ookaze.fr/Xenoblade/

# Convert gif to png (map on website is a single image in a gif)
def map_gif_to_png(filename):
    img = Image.open(filename)
    img.seek(0)
    img.save(filename.removesuffix("gif") + "png")

# filename = "snowmt.gif"
# map_gif_to_png(filename)


# Shows an image but resizing it to the size of my screen
def show(window, img):
    height, width, _ = img.shape
    r1 = 600 / height
    r2 = 1100 / width
    r = min(1, r1, r2)
    cv2.imshow(window, cv2.resize(img, (int(width*r), int(height*r))))

# Trys to mask out the lines of the grid, found houghlinnes to work better
# DO NOT USE
def find_mask(filename):
    '''
    (255, 238, 214) <- Water
    (254, 206, 168) <- good ? 

    Lines touched: 165 265 376 377 378 410 476 765 865 964 1065 1150

    Grid:
    [254, 206, 168] <- Biggg
    [254, 183, 130] <- Biggg
    [225, 189, 162] <- Small
    [255, 189, 177] <- Purple

    
    '''


    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    print(img.shape)

    # 1st mask outside borders
    # low_blue_ext = np.array([250, 200, 165])
    # high_blue_ext = np.array([260, 210, 175])
    # mask_ext = cv2.inRange(img, low_blue_ext, high_blue_ext)
    # for i in range(1200):
    #     if (mask_ext[i][165] == 255):
    #         # print(i, end = " ")
    #         print(img[i][165])

    good_colors = [[254, 206, 168], [254, 183, 130], [225, 189, 162], [255, 189, 177]]
    good_colors_array = [np.array(i) for i in good_colors]
    mask = None
    for color in good_colors_array:
        curr_mask = cv2.inRange(img, color, color)
        if mask is None:
            mask = curr_mask
        else:
            mask = cv2.bitwise_or(mask, curr_mask)


    # low_blue_ext = np.array([250, 200, 177])
    # high_blue_ext = np.array([255, 230, 200])
    # curr_mask = cv2.inRange(img, low_blue_ext, high_blue_ext)
    # mask = curr_mask

    res = cv2.bitwise_and(img, img, mask=mask)
    
    # cv2.imwrite("bnew.png", mask)
    # show("Mask", res)
    # cv2.waitKey(0)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)

    return res



#### FINDING THE LINES ####

# Merges close elements in a a sorted list
def uniform_list(l):
    l.sort()
    for i in range(len(l)-2, -1, -1):
        if(l[i+1] - l[i] < 5):
            l[i] = (l[i] + l[i+1]) / 2
            del(l[i+1])
    return l

# Find median difference between 2 consecutive elements in a sorted list (assuming it's an int)
def find_spacing(l):
    diff_l = [l[i+1] - l[i] for i in range(len(l)-1)]
    # print(diff_l)
    # print(np.median(diff_l))
    median = np.median(diff_l)
    return 10*round(median/10)  #Round to nearest 10s

# Finds the anchor (linear pts with smallest difference to og list ) in a sorted list and its spacing
def anchor(l, spacing):
    for i in range(len(l)-2,-1,-1):
        k = (l[i+1] - l[i]) / spacing
        for j in range(round(k) - 1):
            l.insert(i+1, None)
    n = len(l)
    pts_to_anchor = [i for i in range(n) if l[i] is not None]
    max_i, max_cost = None, None
    for i in pts_to_anchor:
        cost = 0
        for i in range(n):
            if l[i] is not None:
                cost += abs(k-i)*spacing
        if max_cost is None or cost < max_cost:
            max_i = i
    best_pts = list(map(round,[l[max_i] + (k-max_i)*spacing for k in range(n)]))
    return best_pts

# Finds the list of x and y indexes which corresponds to the lines on the image
def find_grid(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    height, width, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)
    v_lines = cv2.HoughLinesP(edges, 1, np.pi, 2, None, 300, 10)
    v_lines = list(filter(lambda x: x[0][0] == x[0][2], v_lines))
    X = uniform_list([line[0][0] for line in v_lines])
    h_lines = cv2.HoughLinesP(edges, 1, np.pi / 2, 10, None, 300, 5)
    h_lines = list(filter(lambda x: x[0][1] == x[0][3], h_lines))
    Y = uniform_list([line[0][1] for line in h_lines])
    spacing_x = find_spacing(X)
    spacing_y = find_spacing(Y)
    # print("Found", len(X), "vertical lines, spacing is", spacing_x)
    # print("Found", len(Y), "horizontal lines, spacing is", spacing_y)
    X = anchor(X, spacing_x)
    Y = anchor(Y, spacing_y)

    # for line in v_lines:
    #     payload = line[0]
    #     pt1 = (payload[0],payload[1])
    #     pt2 = (payload[2],payload[3])
    #     cv2.line(img, pt1, pt2, (0,255,0), 2)
    # show("Grid", img)
    # cv2.waitKey(0)

    return X, spacing_x, Y, spacing_y

# Draws a grid
def draw_grid(filename, X, Y):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    for x in X:
        cv2.line(img, (x, 0), (x, height), (0,255,0), 2)
    for y in Y:
        cv2.line(img, (0, y), (width, y), (0,255,0), 2)
    show("Grid", img)
    cv2.waitKey(0)


#### FINDING THE COORDINATES ####

# Find all (word, (x_c, y_c)) pairs in img
def get_words_with_center(img):
    height, width, _ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    if not boxes:
        return []
    boxes = [i.split(" ") for i in boxes.strip().split("\n")]
    for x in boxes:
        for i in range(1, len(x)):
            x[i] = int(x[i])
        x[2] = height - x[2]
        x[4] = height - x[4]
    strings = pytesseract.image_to_string(img)
    data = []  #List of word / center

    curr_string = None
    i = 0
    for s in strings:
        if s.strip() == "":
            if curr_string is not None:
                curr_center = ((curr_topleft[0] + curr_botright[0])/2, (curr_topleft[1] + curr_botright[1])/2)
                data.append((curr_string, curr_center))
                curr_string = None
        else:
            while s.strip() != boxes[i][0]:
                i += 1
            if curr_string is None:
                curr_string = s
                curr_topleft = (boxes[i][1], boxes[i][2])
                curr_botright = (boxes[i][3], boxes[i][4])
                i += 1
            else:
                curr_string += s
                curr_botright = (boxes[i][3], boxes[i][4])
                i += 1
    
    return data

# Finds the coordinate associated with a vertical line
def find_horizontal_coordinate(filename, x, spacing_x, spacing_y):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    sub_image = img[0:spacing_y, max(0,x-spacing_x):x+spacing_x]
    sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)

    data = get_words_with_center(sub_image_rgb)
    for i in range(len(data)):
        if data[i][0].lower() in ("x:0", "y:0"):
            data[i] = ("0", data[i][1])
    good_data = []
    for (word, (xa, ya)) in data:
        if word.removeprefix('-').isnumeric():
            good_data.append((int(word), abs(round(xa-spacing_x)) + ya))
        cv2.circle(sub_image, (round(xa),round(ya)), 3, (0, 255, 0), -1)
    good_data.sort(key = lambda x: x[1])
    if not good_data:
        return None

    # show("Sub", sub_image)
    # cv2.waitKey(0)
    return good_data[0][0]

# Finds the coordinate associated with a horizontal line
def find_vertical_coordinate(filename, y, spacing_x, spacing_y):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    sub_image = img[max(0,y-spacing_y):y+spacing_y, 0:spacing_x]
    sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)

    data = get_words_with_center(sub_image_rgb)
    for i in range(len(data)):
        if data[i][0].lower() in ("x:0", "y:0"):
            data[i] = ("0", data[i][1])
    good_data = []
    for (word, (xa, ya)) in data:
        if word.removeprefix('-').isnumeric():
            good_data.append((int(word), abs(round(ya-spacing_y)) + xa))
        cv2.circle(sub_image, (round(xa),round(ya)), 3, (0, 255, 0), -1)
    good_data.sort(key = lambda x: x[1])
    if not good_data:
        return None

    # cv2.imshow("Sub", sub_image)
    # cv2.waitKey(0)
    return good_data[0][0]

# Finds the start and step of the scale
def find_scale(filename, coords):
    steps = {}   # Dictionnary of possible steps and how many approved
    starts = {}  # Dictionnary of possible starts and how many approved
    last_seen = coords[0]
    curr_diff = 1
    non_null_elements = 0
    for i in range(1, len(coords)):
        if coords[i] is not None:
            non_null_elements += 1
            step = (coords[i] - last_seen) // curr_diff
            start = coords[i] - i*step
            if step in steps.keys():
                steps[step] += 1
            else:
                steps[step] = 1
            if start in starts.keys():
                starts[start] += 1
            else:
                starts[start] = 1
            last_seen = coords[i]
            curr_diff = 1
        else:
            curr_diff += 1

    good_step = max(steps.items(), key= lambda x: x[1])
    good_start = max(starts.items(), key= lambda x: x[1])
    if (good_step[1] > .6*non_null_elements and good_start[1] > .6*non_null_elements):
        return (good_start[0], good_step[0])
    else:
        print("Starts:", starts, "\nSteps:", steps)
        raise(Exception("Not every line is saying the same thing"))

    

#### FINDING THE TRANSITION FUNCTION ####

# Finds the transition coefficients of an image: (a, b, c, d) so that (x, y) -> (ax+b, cy+d)
def find_transition_coefs(filename):
    X, spacing_x, Y, spacing_y = find_grid(filename)
    x_coords = [find_horizontal_coordinate(filename, x, spacing_x, spacing_y) for x in X]
    x_start, x_spacing_step = find_scale(filename, x_coords)
    y_coords = [find_vertical_coordinate(filename, y, spacing_x, spacing_y) for y in Y]
    y_start, y_spacing_step = find_scale(filename, y_coords)

    x_step = x_spacing_step/spacing_x
    y_step = y_spacing_step/spacing_y
    x_0 = x_start-X[0]*x_step
    y_0 = y_start-Y[0]*y_step

    return (x_step, x_0, y_step, y_0)

'''
+ spacing_x (pixels) <=> + x_step (coords)
x = 0 (pixels) <=> x = x_0 (coords) 
'''
    

#### DRAWABLE MAP ####

class DrawableMap():
    def __init__(self, filename):
        self.filename = filename
        self.map = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.map_title = f"Map of {self.filename.removesuffix('.png')}"
        self.height, self.width, _ = (self.map).shape
        self.x_step, self.x_0, self.y_step, self.y_0 = find_transition_coefs(filename)

    def pixels_to_coords(self, pt):
        xp, yp = pt
        xc = self.x_step*xp + self.x_0
        yc = self.y_step*yp + self.y_0
        return (round(xc), round(yc))

    def coords_to_pixels(self, pt):
        x, y = pt
        return (round((x-self.x_0)*self.x_step), round((y-self.y_0)*self.y_step))

    def place_point(self, coords, radius, color, thickness):
        cv2.circle(self.map, self.coords_to_pixels(coords), radius, color, thickness)
    
    def show(self):
        show(self.map_title, self.map)
        cv2.waitKey(0)
        cv2.destroyWindow(self.map_title)





filename = "snowmt.png"

colony9_map = DrawableMap(filename)
# point = (1000, 700)
# cv2.circle(colony9_map.map, point, 10, (0,255,0), -1)
# print(colony9_map.pixels_to_coords(point))
colony9_map.place_point((-800, 700), 10, (0, 255, 0), -1)
colony9_map.show()
