#!/usr/bin/env python

import rospy 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64



x=0.0

def callback(data):
	global x
	if data.position[0]!=0:
		x=data.position[0]
	print(x)


def rotate(angle):
	speed=Float64()
	sub = rospy.Subscriber("/catapult/joint_states", JointState, callback)
	pub = rospy.Publisher("/catapult/base_rotation_controller/command", Float64, queue_size=1)




	while True:
		inc_theta=angle-x
		print(inc_theta)
		if abs(inc_theta)>0.05:
			speed.data=0.3*inc_theta
			pub.publish(speed)

		if abs(inc_theta)<0.05:
			speed.data=0
			pub.publish(speed)
			break

	rospy.sleep()







def publisher():
	rospy.init_node('umic_bot1',anonymous=True)
	pub = rospy.Publisher("/catapult/base_rotation_controller/command", Float64, queue_size=1)
	sub = rospy.Subscriber("/catapult/joint_states", JointState, callback)
	r=rospy.Rate(100)


	while not rospy.is_shutdown():
		rotate(2)
		break

	

	
	



if __name__=='__main__':
	try:
		publisher()

		
	except rospy.ROSInterruptException:
		pass 

