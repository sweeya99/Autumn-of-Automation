#!/usr/bin/env python
from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import rospy
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from math import atan2
import math
import time
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry

pixel =64
#global alphabets 
alphabets = ['P', 'S', 'L', 'U', 'B', 'O', 'G', 'N', 'C', 'Y', 'K', 'C', 'M', 'F', 'I', 'H', 'A', 'T']

global c
global out

rospy.init_node('umic_bot',anonymous=True)
#pub=rospy.Publisher('/mybot/mobile_base_controller/cmd_vel',Twist,queue_size=10)
#pub2=rospy.Publisher('/mybot/gripper_extension_controller/command',Float64,queue_size=10)


c=Float64()
out=Twist()

global PI
PI = 3.1415926535897

x = 0.0
y = 0.0
#z = 0.0
theta = 0.0
global kp
kp = 0.5
t=0
v=0


def newOdom(msg):
    global x
    global y
    #global z
    global theta
    global v         # v = angular speed (x)
    global t         # t = angular speed(z)

    x = msg.pose.pose.position.x  # pose.position.x
    y = msg.pose.pose.position.y  # pose.position.y
    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion(
        [rot_q.x, rot_q.y, rot_q.z, rot_q.w])


    v = msg .twist.twist.linear.x
    t = msg .twist.twist.angular.z

def distance(x1,y1,x2,y2):
	d=math.sqrt(((x2-x1)**2)+((y2-y1)**2))

	return d



def path_plan(px, py):

    
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
    

    speed = Twist()

    r = rospy.Rate(10)
    
    goal = Point()
    goal.x = px
    goal.y = py

    while not rospy.is_shutdown():

        inc_x = goal.x - x
        inc_y = goal.y - y

        print("goal_x={}  goal_y:{}".format(goal.x,goal.y))
        print("current_x={}  current_y:{}".format(x,y))
        print("inc_x={}  inc_y:{}".format(inc_x,inc_y))

        

        angle_to_goal = atan2(inc_y, inc_x)
        print("Target_angle={}  Current_theta:{}".format(angle_to_goal,theta))
        print(abs(angle_to_goal - theta))

        if abs(angle_to_goal - theta) > 0.1:      #0.01
            
            print("1")
            speed.linear.x = 0.0
            #speed.angular.z = 0.3
            speed.angular.z =((angle_to_goal)- theta)*0.3
            print("linearspeed={}  angularspeed:{}".format(v ,t))
            '''
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                print("2")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break

            '''
            if abs(inc_x )<0.05 and abs(inc_y) <0.05 :
                print("2")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                print("v={}  w:{}".format(v ,t))
                print("linearspeed1={}  angularspeed1:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                print("v={}  w:{}".format(v ,t))
                print("linearspeed2={}  angularspeed2:{}".format(speed.linear.x ,speed.angular.z))
                r.sleep()
                print("v={}  w:{}".format(v ,t))
                print("linearspeed3={}  angularspeed3:{}".format(speed.linear.x ,speed.angular.z))
                break
            
              
        else:
            
            
            
            '''
            if inc_x >0.05 and inc_y >0.05 :

                print("3")
                speed.linear.x = 0.2
                speed.angular.z = 0.0
                pub.publish(speed)
                r.sleep()
                

            else :
                print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                pub.publish(speed)
                r.sleep()
                break
            
            '''

            print("3")
            speed.linear.x = 0.2
            speed.angular.z = 0.0


            print("v={}  w:{}".format(v ,t))
            print("linearspeed4={}  angularspeed4:{}".format(speed.linear.x ,speed.angular.z))
            '''
            if inc_x <0.01 and inc_y <0.01 :
                print("4")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break
             '''   
            
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 

                print("v={}  w:{}".format(v ,t))
                print("linearspeed5={}  angularspeed5:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                r.sleep()
                break
                

        pub.publish(speed)
        
        print("confirm")
        r.sleep()
        print("not stopping") 

def orient_along(px, py):


    #rospy.init_node("speed_controller")

    #sub = rospy.Subscriber("/mybot/mobile_base_controller/odom", Odometry, newOdom)
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)

    speed = Twist()

    goal = Point()
    ##r = rospy.Rate(1000)
    goal.x = px
    goal.y = py
    #print(goal.x)
    #print(goal.y)
    while True:

        inc_x = goal.x - x
        inc_y = goal.y - y
        #print(x,y)
        #print(inc_x,inc_y)
        
        
        angle_to_goal = atan2(inc_y, inc_x)
        #angle_to_goal_rad = angle_to_goal * (math.pi/180)
        #print(angle_to_goal)
        #print(theta)
        #print(abs(angle_to_goal - theta))
        if abs(angle_to_goal - theta) > 0.01:
            

            #speed.angular.z =0.9
            speed.angular.z =((angle_to_goal)- theta)*0.3
            
        
        else:
        #    speed.linear.x = 0.0
            speed.angular.z = 0.0
            #pub.publish(speed)
            #r.sleep()
            break
            
                
        pub.publish(speed)
        print("Target={}  Current:{}".format(angle_to_goal,theta))
        r.sleep()
        print("not stopping")   

def point_decide(bx,by,tx,ty):
	angle_to_goal = atan2(ty-by,tx-bx)
	xf=bx-0.28*cos(angle_to_goal)
	yf=by-0.28*sin(angle_to_goal)
	return (xf,yf)

def decide_piston_speed(bx,by,tx,ty):
	d=distance(bx,by,tx,ty)

	v= math.sqrt(55*d/4)

	return v


            

def publisher():
    global r
    r = rospy.Rate(100)
    c.data=0
    pub2=rospy.Publisher('/mybot/gripper_extension_controller/command',Float64,queue_size=10)


    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0
    t0=0
    t1=0


    while not rospy.is_shutdown():

        path_plan(2.5,-3)
        path_plan(2.5,3)



            
        print('done')
        break


if __name__=='__main__':
	try:
		publisher()

		
	except rospy.ROSInterruptException:
		pass      
