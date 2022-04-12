# ---------------------------------------------------------------------------------------------------------------------
#     click - draw line
#SHT click move all
#CTL click move current
# ---------------------------------------------------------------------------------------------------------------------
import numpy
import cv2
from numpy import unravel_index
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_image
import tools_draw_numpy
from CV import tools_alg_match
# ---------------------------------------------------------------------------------------------------------------------
from utils_primities import Boxes,Lines
# ---------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------------------------------------------------------------------
folder_in = './images/ex_KZ/'
filename_boxes = 'df_boxes_filtered.csv'
filename_lines = None
sep = ','
idx_ID,idx_pos=0,3
# ---------------------------------------------------------------------------------------------------------------------
class markup:

    def __init__(self, folder_in,filename_boxes=None,filename_lines=None,sep=' ',idx_ID=0,idx_pos=1):

        self.scale = 1.0
        self.draw_labels = False
        self.folder_in = folder_in
        self.textcolor = (0, 0, 0)
        self.memorized = []

        self.w = 2
        self.image_small = None

        self.temp_line = numpy.zeros(4,dtype=numpy.int)

        self.filenames = tools_IO.get_filenames(self.folder_in, '*.jpg,*.png')
        self.filenames = numpy.unique(self.filenames)
        self.dict_filenames = {x: i for i, x in enumerate(self.filenames)}

        self.current_frame_ID=0
        self.image_to_display = None
        self.current_markup_ID = 0
        self.last_insert_size = 1

        if filename_lines is not None:
            self.display_primitives = 'L'
            self.filename_lines = filename_lines
            self.Lines = Lines()
            self.Lines.read_from_file(self.folder_in + filename_lines,self.dict_filenames)
            self.class_names_lines = self.init_class_names_lines()
            self.colors_lines = tools_draw_numpy.get_colors(len(self.class_names_lines),shuffle=True)
        elif filename_boxes is not None:
            self.display_primitives = 'B'
            self.filename_boxes = filename_boxes
            self.Boxes = Boxes(self.folder_in + filename_boxes,self.dict_filenames,sep=sep,idx_ID=idx_ID,idx_pos=idx_pos)
            self.class_names_boxes = self.init_class_names_boxes()
            self.colors_boxes = tools_draw_numpy.get_colors(len(self.class_names_boxes))
        else:
            self.display_primitives = 'X'
            self.Lines = None
            self.Boxes = None

        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def init_class_names_lines(self):
        lst = ['obj_%d'%i for i in range(16)]
        return lst
# ---------------------------------------------------------------------------------------------------------------------
    def init_class_names_boxes(self):
        lst = ['%d' % i for i in range(25)]
        return lst
# ---------------------------------------------------------------------------------------------------------------------
    def goto_next_frame(self):
        self.current_frame_ID+=1
        if self.current_frame_ID>=len(self.filenames):self.current_frame_ID = 0
        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def goto_prev_frame(self):
        self.current_frame_ID-=1
        if self.current_frame_ID<0:self.current_frame_ID = len(self.filenames) - 1
        self.refresh_image()
        return
# ---------------------------------------------------------------------------------------------------------------------
    def draw_legend(self, image):
        if self.textcolor == (0, 0, 0):
            thikness = 1
        else:
            thikness = 2

        scale = 0.5
        cv2.rectangle(image, (0, 0), (300, 48), (128, 128, 128), -1)
        cv2.putText(image, '  {0:d} {1:s}'.format(self.current_frame_ID, self.filenames[self.current_frame_ID]), (2, 16),cv2.FONT_HERSHEY_SIMPLEX, scale, self.textcolor, thikness, cv2.LINE_AA)

        if self.display_primitives in ['L','S']:
            color = self.colors_lines[self.current_markup_ID]
            class_name = self.class_names_lines[self.current_markup_ID]
            cv2.putText(image, '{0:s} {1:d} {2:s}'.format(self.display_primitives, self.current_markup_ID, class_name), (2, 36),cv2.FONT_HERSHEY_SIMPLEX, scale, color.tolist(), 1, cv2.LINE_AA)

        if self.display_primitives in ('B'):
            color = self.colors_boxes[self.current_markup_ID]
            class_name = self.class_names_boxes[self.current_markup_ID]
            cv2.putText(image, '{0:s} {1:d} {2:s}'.format(self.display_primitives, self.current_markup_ID, class_name), (2, 36),cv2.FONT_HERSHEY_SIMPLEX, scale, color.tolist(), 1, cv2.LINE_AA)

        return image

# ---------------------------------------------------------------------------------------------------------------------
    def refresh_image(self):
        self.image_to_display = cv2.imread(self.folder_in + self.filenames[self.current_frame_ID])
        self.image_to_display = tools_image.desaturate(self.image_to_display, level=0.75)

        if self.display_primitives in ('L'):
            self.image_to_display = self.draw_lines(self.image_to_display)
            self.image_to_display = self.draw_segments(self.image_to_display,draw_labels=self.draw_labels)
        if self.display_primitives in ('B'): self.image_to_display = self.draw_boxes(self.image_to_display)
        if self.display_primitives in ('S'):
            self.image_to_display = self.draw_lines(self.image_to_display)
            self.image_to_display = self.draw_segments(self.image_to_display,draw_labels=self.draw_labels)


        self.image_to_display = self.draw_legend(self.image_to_display)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_next_frame_by_marked_line(self, look_forward=True, same_class=False):

        start, success = self.current_frame_ID, False
        current = self.current_frame_ID

        while not success:
            if look_forward:
                current += 1
                if current >= len(self.filenames): current = 0
            else:
                current -= 1
                if current < 0: current = len(self.filenames) - 1

            if current in self.Lines.filename_IDs:
                if not same_class:
                    success = True
                else:
                    idx = numpy.where(numpy.array(self.Lines.filename_IDs) == current)
                    labels = numpy.array(self.Lines.ids)[idx]
                    if self.current_markup_ID in labels:
                        success = True
            if current == start:
                success = True

        return current
# ---------------------------------------------------------------------------------------------------------------------
    def get_next_frame_by_marked_box(self, look_forward=True, same_class=False):

        start, success = self.current_frame_ID, False
        current = self.current_frame_ID

        while not success:
            if look_forward:
                current += 1
                if current >= len(self.filenames): current = 0
            else:
                current -= 1
                if current < 0: current = len(self.filenames) - 1

            if current in self.Boxes.filename_IDs:
                if not same_class:
                    success = True
                else:
                    idx = numpy.where(numpy.array(self.Boxes.filename_IDs) == current)
                    labels = numpy.array(self.Boxes.ids)[idx]
                    if self.current_markup_ID in labels:
                        success = True
            if current == start:
                success = True

        return current

# ---------------------------------------------------------------------------------------------------------------------
    def extent_to_next_frame(self,next_frame,all=True):

        def get_objects(frame_ID, classID=None):
            idx = numpy.where(self.Lines.filename_IDs == frame_ID)
            classIDs = self.Lines.ids[idx]
            lines = self.Lines.xyxy[idx]
            if classID is None: return lines, classIDs
            else:
                idx = numpy.where(classIDs == classID)
                return lines[idx], classIDs[idx]

        if all==True:
            lines_current, classIDs = get_objects(self.current_frame_ID)
        else:
            lines_current, classIDs = get_objects(self.current_frame_ID, self.current_markup_ID)

        next_classIDs = numpy.unique(self.Lines.ids[numpy.where(self.Lines.filename_IDs == next_frame)])

        for line,clsssID in zip(lines_current,classIDs):
            if clsssID in next_classIDs: continue
            self.Lines.add(next_frame,line,clsssID,do_standartize=False)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def draw_lines(self, image, draw_labels=True):

        image = cv2.line(image, (self.temp_line[0], self.temp_line[1]), (self.temp_line[2], self.temp_line[3]),(255, 255, 0), self.w)

        for i, each in enumerate(self.Lines.filename_IDs):
            if each == self.current_frame_ID:
                id = self.Lines.ids[i]
                name = self.class_names_lines[int(id)]
                left,top,right,bottom = self.Lines.xyxy[i]
                cv2.line(image, (left, top), (right, bottom), self.colors_lines[id].tolist(), self.w)
                #if draw_labels:
                    #cv2.putText(image, '{0:d}'.format(id), ((left+right)//2,(top+bottom)//2),cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[id], 1, cv2.LINE_AA)
                    #cv2.putText(image, '{0}'.format(name), ((left+right)//2,(top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors_lines[id].tolist(), 1, cv2.LINE_AA)



        return image
# ---------------------------------------------------------------------------------------------------------------------
    def draw_segments(self, image, draw_labels=True):

        lines = self.Lines.get(self.current_frame_ID)
        ids = numpy.unique(lines.ids)
        for id in ids:
            pnts = lines.xyxy[lines.ids == id].reshape((-1,2))
            image = tools_draw_numpy.draw_contours(image, pnts, self.colors_lines[id].tolist(), transperency=0.30)
            if draw_labels:
                cv2.putText(image, '{0:d}'.format(id), (int(numpy.mean(pnts[:,0])),int(numpy.mean(pnts[:,1]))),cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors_lines[id].tolist(), 1, cv2.LINE_AA)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def add_line(self):
        left   = int(g_coord[0][0]/self.scale)
        top    = int(g_coord[0][1]/self.scale)
        right  = int(g_coord[1][0]/self.scale)
        bottom = int(g_coord[1][1]/self.scale)
        self.Lines.add(self.current_frame_ID,(left,top,right,bottom),self.current_markup_ID,do_standartize=False)
        self.last_insert_size = 1
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_line(self):
        top    = g_coord[0][1]/self.scale
        left   = g_coord[0][0]/self.scale
        bottom = g_coord[1][1]/self.scale
        right  = g_coord[1][0]/self.scale
        self.Lines.remove_by_cut(self.current_frame_ID, (left,top,right,bottom))
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_last_line(self, N=1):
        if N==1:
            M.Lines.remove_last()
            return

        if N>1 and len(M.Lines.ids)>=N:
            M.Lines.remove_last()

        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_all_from_current_frame(self):
        if self.display_primitives in ('L'):
            self.Lines.remove_by_fileID(self.current_frame_ID)

        if self.display_primitives in ('B'):
            self.Boxes.remove_by_fileID(self.current_frame_ID)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def shift_objects(self, delta,do_all, memorized=None):

        if do_all:
            idx = numpy.where(self.Lines.filename_IDs == self.current_frame_ID)
        else:
            f1 = numpy.array(self.Lines.filename_IDs == self.current_frame_ID)
            f2 = numpy.array(self.Lines.ids == self.current_markup_ID)
            idx = numpy.where(f1 * f2)

        if memorized is None:
            memorized = self.Lines.xyxy[idx]

        self.Lines.xyxy[idx]= memorized + numpy.array((delta[0], delta[1], delta[0], delta[1]))

        return
# ---------------------------------------------------------------------------------------------------------------------
    def shift_objects_remember(self,all=True):

        if all:
            idx = numpy.where(self.Lines.filename_IDs == self.current_frame_ID)
        else:
            f1 = numpy.array(self.Lines.filename_IDs == self.current_frame_ID)
            f2 = numpy.array(self.Lines.ids == self.current_markup_ID)
            idx = numpy.where(f1 * f2)

        momorized = self.Lines.xyxy[idx]
        return momorized
# ---------------------------------------------------------------------------------------------------------------------
    def draw_boxes(self, image):
        for i, each in enumerate(self.Boxes.filename_IDs):
            if each == self.current_frame_ID:
                id = self.Boxes.ids[i]
                name = self.class_names_boxes[id]
                left, top, right, bottom = self.Boxes.xyxy[i].astype(int)
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), self.colors_boxes[id].tolist(), self.w)
                if self.draw_labels:
                    cv2.putText(image, '{0}'.format(name), ((left + right) // 2, (top) ),cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors_boxes[id].tolist(), 1, cv2.LINE_AA)

        return image
# ---------------------------------------------------------------------------------------------------------------------
    def get_box_objects(self,frame_ID, classID=None):
        idx = numpy.where(self.Boxes.filename_IDs == frame_ID)
        classIDs = self.Boxes.ids[idx]
        boxes = self.Boxes.xyxy[idx]
        if classID is None:
            return boxes, classIDs
        else:
            idx = numpy.where(classIDs == classID)
            return boxes[idx], classIDs[idx]
# ---------------------------------------------------------------------------------------------------------------------
    def add_boxes(self):

        next_frame_ID = self.get_next_frame_by_marked_box(look_forward=True, same_class=True)
        if next_frame_ID <= self.current_frame_ID: return
        if self.current_frame_ID + 1 == next_frame_ID: return

        box_current, classIDs = self.get_box_objects(self.current_frame_ID, self.current_markup_ID)
        box_next, classIDs = self.get_box_objects(next_frame_ID, self.current_markup_ID)

        if not (len(box_current) == 1 and len(box_next) == 1): return

        box_start = numpy.array(box_current[0]).astype(float)
        box_end = numpy.array(box_next[0]).astype(float)

        for frameID in range(self.current_frame_ID + 1, next_frame_ID):
            alpha = (frameID - self.current_frame_ID) / (next_frame_ID - self.current_frame_ID)
            box = box_start * (1 - alpha) + box_end * (alpha)
            self.Boxes.add(frameID, box, self.current_markup_ID)

        self.last_insert_size = next_frame_ID - self.current_frame_ID - 1
        self.current_frame_ID = next_frame_ID-1

        return

# ---------------------------------------------------------------------------------------------------------------------
    def add_box_automated(self):
        box_current, classIDs = self.get_box_objects(self.current_frame_ID, self.current_markup_ID)
        if len(box_current) != 1: return
        box = numpy.array(box_current[0]).astype(int)

        image_current = cv2.imread(self.folder_in + self.filenames[self.current_frame_ID])

        cur_left = min(box[0],box[2])
        cur_right = max(box[0], box[2])
        cur_top = min(box[1], box[3])
        cur_bottom  = max(box[1], box[3])

        R = 100
        next_left = max(0,cur_left - R)
        next_right = min(cur_right + R,image_current.shape[1])
        next_top = max(0,cur_top - R)
        next_bottom = min(cur_bottom + R,image_current.shape[0])

        if self.image_small is None:
            self.image_small = image_current[cur_top:cur_bottom,cur_left:cur_right]

        image_next = cv2.imread(self.folder_in + self.filenames[self.current_frame_ID+1])
        image_large = image_next[next_top:next_bottom,next_left:next_right]

        hitmap = tools_alg_match.calc_hit_field(image_large,self.image_small)
        hit = unravel_index(hitmap.argmax(), hitmap.shape)

        box = numpy.array([next_left+hit[1]-(cur_right-cur_left)//2,
                           next_top +hit[0]-(cur_bottom-cur_top)//2,

                           next_left+hit[1]+(cur_right-cur_left)//2,
                           next_top +hit[0]+(cur_bottom-cur_top)//2
                           ])

        self.Boxes.add(self.current_frame_ID+1, box, self.current_markup_ID)
        #image_small_next = image_next[box[1]:box[3], box[0]:box[2]]
        #cv2.imwrite(self.folder_in + 'hit.png',tools_image.hitmap2d_to_jet(hitmap))
        #cv2.imwrite(self.folder_in + 'small.png', image_small)
        #cv2.imwrite(self.folder_in + 'large.png', image_large)
        #cv2.imwrite(self.folder_in + 'image_small_next.png', image_small_next)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def add_box(self):
        left   = int(g_coord[0][0]/self.scale)
        top    = int(g_coord[0][1]/self.scale)
        right  = int(g_coord[1][0]/self.scale)
        bottom = int(g_coord[1][1]/self.scale)
        self.Boxes.add(self.current_frame_ID,(left,top,right,bottom),self.current_markup_ID)
        self.last_insert_size = 1
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_box_by_point(self):
        x = g_coord[0][0] / self.scale
        y = g_coord[0][1]/self.scale
        #bottom = g_coord[1][1]/self.scale
        #right  = g_coord[1][0]/self.scale
        self.Boxes.remove_by_point(self.current_frame_ID, (x,y))
        return
# ---------------------------------------------------------------------------------------------------------------------
    def remove_box_by_line(self):
        left   = g_coord[0][0]/self.scale
        top    = g_coord[0][1]/self.scale
        right = g_coord[1][0] / self.scale
        bottom = g_coord[1][1]/self.scale

        self.Boxes.remove_by_cut(self.current_frame_ID, (left,top,right,bottom))
        return
# =====================================================================================================================
g_coord, g_current_mouse_pos, g_mouse_event_code = [], numpy.zeros(4,dtype=numpy.int), None
M = markup(folder_in,filename_boxes=filename_boxes,filename_lines=filename_lines,sep=sep,idx_ID=idx_ID,idx_pos=idx_pos)
# ---------------------------------------------------------------------------------------------------------------------
def click_handler(event, x, y, flags, param):

    global g_coord, g_current_mouse_pos, g_mouse_event_code

    is_ctrl  = (flags&0x08)>0
    is_shift = (flags&0x10)>0
    is_alt   = (flags&0x20)>0

    g_current_mouse_pos = (x,y)

    if g_mouse_event_code == 'LBUTTONDOWN':
        M.temp_line[2] = g_current_mouse_pos[0]
        M.temp_line[3] = g_current_mouse_pos[1]
        M.refresh_image()
    else:
        M.temp_line = numpy.zeros(4,dtype=int)

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (is_shift):
        g_coord.append((x, y))
        M.memorized = M.shift_objects_remember(all=True)
        g_mouse_event_code = 'LBUTTONDOWN_SHF'

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (is_ctrl):
        g_coord.append((x, y))
        M.memorized = M.shift_objects_remember(all=False)
        g_mouse_event_code = 'LBUTTONDOWN_CTL'

    if event == cv2.EVENT_LBUTTONUP and g_mouse_event_code == 'LBUTTONDOWN_SHF':
        g_coord.append((x, y))
        g_mouse_event_code = 'LBUTTONUP_SHF'

    if event == cv2.EVENT_LBUTTONUP and g_mouse_event_code == 'LBUTTONDOWN_CTL':
        g_coord.append((x, y))
        g_mouse_event_code = 'LBUTTONUP_CTL'

    if event == cv2.EVENT_LBUTTONDOWN and g_mouse_event_code is None and (not is_shift) and (not is_ctrl):
        g_coord.append((x, y))
        M.temp_line[0] = g_coord[0][0]
        M.temp_line[1] = g_coord[0][1]
        g_mouse_event_code = 'LBUTTONDOWN'

    elif event == cv2.EVENT_LBUTTONUP and g_mouse_event_code=='LBUTTONDOWN':
        g_coord.append((x, y))
        g_mouse_event_code = 'LBUTTONUP'

    elif event == cv2.EVENT_RBUTTONDBLCLK and g_mouse_event_code is None:
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONDBLCLK'

    elif event == cv2.EVENT_RBUTTONDOWN and g_mouse_event_code is None:
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONDOWN'

    elif event == cv2.EVENT_RBUTTONUP and g_mouse_event_code=='RBUTTONDOWN':
        g_coord.append((x, y))
        g_mouse_event_code = 'RBUTTONUP'

    if event == cv2.EVENT_MOUSEWHEEL:
        if is_ctrl:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_CTL_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_CTL_BK'
        elif is_shift:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_SHF_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_SHF_BK'
        elif is_alt:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_ALT_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_ALT_BK'
        else:
            if flags>0:g_mouse_event_code = 'MOUSEWHEEL_FW'
            else:g_mouse_event_code = 'MOUSEWHEEL_BK'

    return
# ---------------------------------------------------------------------------------------------------------------------
def process_mouse():
    global g_coord, g_current_mouse_pos, g_mouse_event_code

    if g_mouse_event_code == 'LBUTTONDOWN':
        return 1

    # TODO - shift
    if g_mouse_event_code == 'LBUTTONDOWN_CTL':
        M.shift_objects(numpy.array(g_current_mouse_pos)-numpy.array(g_coord[0]),do_all=False,memorized=M.memorized)
        M.refresh_image()
        return 1

    if g_mouse_event_code == 'LBUTTONDOWN_SHF':
        M.shift_objects(numpy.array(g_current_mouse_pos)-numpy.array(g_coord[0]),do_all=True,memorized=M.memorized)
        M.refresh_image()
        return 1

    # TODO - shift end
    if g_mouse_event_code == 'LBUTTONUP_CTL':
        M.shift_objects(numpy.array(g_coord[1])-numpy.array(g_coord[0]),do_all=False,memorized=M.memorized)
        g_coord.pop()
        g_coord.pop()
        M.refresh_image()
        g_mouse_event_code = None
        return 1

    # TODO - shift end
    if g_mouse_event_code == 'LBUTTONUP_SHF':
        M.shift_objects(numpy.array(g_coord[1])-numpy.array(g_coord[0]),do_all=True,memorized=M.memorized)
        g_coord.pop()
        g_coord.pop()
        M.refresh_image()
        g_mouse_event_code = None
        return 1

    # TODO - goto next frame
    if g_mouse_event_code == 'MOUSEWHEEL_FW':
        g_mouse_event_code = None
        M.goto_prev_frame()
        return 1

    # TODO - goto prev frame
    if g_mouse_event_code == 'MOUSEWHEEL_BK':
        M.goto_next_frame()
        g_mouse_event_code = None
        return 1

    # TODO - add item
    if g_mouse_event_code == 'LBUTTONUP':
        if M.display_primitives=='L':
            M.add_line()
            M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)

        elif M.display_primitives == 'B':
            M.add_box()
            M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)

        g_mouse_event_code = None
        g_coord.pop()
        g_coord.pop()
        M.refresh_image()
        return 1

    # TODO - remove line
    if g_mouse_event_code == 'RBUTTONUP':
        if M.display_primitives=='L':
            M.remove_line()
            M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)

        elif M.display_primitives=='B':
            M.remove_box_by_line()
            M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)

        g_coord.pop()
        g_coord.pop()
        g_mouse_event_code = None
        M.refresh_image()
        return 1

    # TODO - remove box
    if g_mouse_event_code == 'RBUTTONDBLCLK':
        if M.display_primitives == 'B':
            M.remove_box_by_point()
            M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)
        g_coord.pop()
        g_mouse_event_code = None
        M.refresh_image()


    # TODO - goto prev marked frame
    if g_mouse_event_code in ['MOUSEWHEEL_CTL_FW','MOUSEWHEEL_SHF_FW']:
        if M.display_primitives == 'L':M.current_frame_ID = M.get_next_frame_by_marked_line(look_forward=False,same_class=True)
        if M.display_primitives == 'B':M.current_frame_ID = M.get_next_frame_by_marked_box(look_forward=False,same_class=True)
        g_mouse_event_code = None
        M.refresh_image()
        return 1

    # TODO - goto next marked frame
    if g_mouse_event_code in ['MOUSEWHEEL_CTL_BK','MOUSEWHEEL_SHF_BK']:
        if M.display_primitives == 'L':M.current_frame_ID = M.get_next_frame_by_marked_line(look_forward=True,same_class=True)
        if M.display_primitives == 'B':M.current_frame_ID = M.get_next_frame_by_marked_box(look_forward=True,same_class=True)
        g_mouse_event_code = None
        M.refresh_image()
        return 1

    # TODO - copy labelling next frame
    if g_mouse_event_code == 'MOUSEWHEEL_ALT_BK':
        if M.display_primitives == 'L':
            M.extent_to_next_frame(M.current_frame_ID+1)
            M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)
            M.goto_next_frame()

        if M.display_primitives == 'B':
            M.add_boxes()
            M.goto_next_frame()
            M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)

        g_mouse_event_code = None
        M.refresh_image()
        return 1


    if g_mouse_event_code == 'MOUSEWHEEL_ALT_FW':
        if M.display_primitives == 'L':
            M.extent_to_next_frame(M.current_frame_ID-1)
            M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)
            M.goto_prev_frame()

        g_mouse_event_code = None
        M.refresh_image()
        return 1


    return 0
# ---------------------------------------------------------------------------------------------------------------------
def process_key(key):
    if key & 0xFF == 27: return -1
    if key & 0xFF == ord('x'):
        if M.display_primitives == 'L':M.current_markup_ID = (M.current_markup_ID + 1) % len(M.colors_lines)
        if M.display_primitives == 'B':M.current_markup_ID = (M.current_markup_ID + 1) % len(M.colors_boxes)

        M.refresh_image()
        return 1

    if key & 0xFF == ord('z'):
        if M.display_primitives == 'L':M.current_markup_ID = (M.current_markup_ID - 1) % len(M.colors_lines)
        if M.display_primitives == 'B':M.current_markup_ID = (M.current_markup_ID - 1) % len(M.colors_boxes)

        M.refresh_image()
        return 1

    if key & 0xFF == ord('i'):
        M.draw_labels = not M.draw_labels
        M.refresh_image()
        return 1


    if key & 0xFF >= ord('0') and key & 0xFF <= ord('9'):
        M.current_markup_ID = (key & 0xFF) - ord('0')
        M.refresh_image()
        return 1

    if key == 26 and M.display_primitives in ['L','B']:
        M.remove_last_line(M.last_insert_size)
        M.refresh_image()
        if M.Lines is not None:M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)
        if M.Boxes is not None:M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)
        return 1

    if key == 0:
        M.remove_all_from_current_frame()
        M.refresh_image()
        if M.Lines is not None: M.Lines.save_to_file(M.folder_in + M.filename_lines, M.filenames)
        if M.Boxes is not None: M.Boxes.save_to_file(M.folder_in + M.filename_boxes, M.filenames)

        return 1

    return 0
# ---------------------------------------------------------------------------------------------------------------------
def application_loop_lines():

    should_be_closed = False
    should_be_refreshed = True

    while not should_be_closed:

        res = process_mouse()
        if res>0:should_be_refreshed |= True

        if should_be_refreshed:
            resized = cv2.resize(M.image_to_display,(int(M.scale*M.image_to_display.shape[1]),int(M.scale*M.image_to_display.shape[0])))
            cv2.imshow(window_name, resized)
            should_be_refreshed = False

        key = cv2.waitKey(1)
        res = process_key(key)
        if res < 0: should_be_closed = True
        if res > 0: should_be_refreshed |= True

    cv2.destroyAllWindows()
    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    window_name = 'image_markup'
    cv2.namedWindow(window_name,cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, click_handler)
    application_loop_lines()

