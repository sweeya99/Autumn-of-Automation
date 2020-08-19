#! /usr/bin/env python

import rospy
import math
from converter.msg import quaternion ,roll_pitch_yaw
from math import atan2

#from tf.transformations import euler_from_quaternion

#def quat2euler(x,y,z,w):
#    quat = [x,y,z,w]
#    return euler_from_quaternion(quat)

class Euler_angles :
    
    def __init__(self, roll, pitch ,yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw



def converter (msg) :
    global x
    global y
    global z
    global w

    x = msg.x
    y = msg.y
    z = msg.z
    w = msg.w
    euler_angles =  Euler_angles()

    #roll
    sinr_cosp = 2 *( ( w * x )+ ( y * z) )
    cosr_cosp = 1 - 2 * (( x * x ) + ( y * y))

    euler_angles.roll = atan2(sinr_cosp ,cosr_cosp)

    #pitch

    sinp = 2* ( ( w * y)-( z * x ) )
    if (abs(sinp)) >=1 :
        euler_angles.pitch = copysign(M_PI /2 ,sinp)
    else :
        euler_angles.pitch = asin(sinp)


    #yaw
    siny_cosp = 2 * (( w * z )+ ( x * y ) )
    cosy_cosp = 1 - 2* (( y * y ) + ( z * z))
    euler_angles.yaw = atan2(siny_cosp ,cosy_cosp)
    '''
    while not rospy.is_shutdown():
        pub = rospy.Publisher('topic2',rollpitchyaw ,queue_size = 10)
        pub.Publish(rpy)
        rate.sleep()
    '''
    #instead of return we need to put the publish commands
    return euler_angles


    #(roll ,pitch ,yaw)=euler_from_quaternion([x ,y ,z ,w ])

def main():

    rospy.init_node("my_converter")
    sub = rospy.Subscriber("/topic1/",quaternion ,converter)
    pub = rospy.Publisher("/topic2/",roll_pitch_yaw,queue_size=1)   
    r= rospy.Rate(4) 

    msg_value= roll_pitch_yaw()   
                  #msg_value nothing but message of publisher
    while not rospy.is_shutdown():
        
        msg_value.roll  = 3         #input("enter value of roll :")
        msg_value.pitch = 3         #input("enter value of pitch :")
        msg_value.yaw   = 3         #input("enter value of yaw  :")
        
        pub.publish(msg_value)
        r.sleep()



if __name__=="__main__":
    try:
        main()
    
    except rospy.ROSInterruptException:
        pass