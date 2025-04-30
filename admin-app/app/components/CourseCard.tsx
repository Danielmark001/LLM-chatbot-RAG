"use client";

import React,{useState} from 'react';
import { IoIosCloseCircleOutline } from "react-icons/io";
import Link from 'next/link';

interface CourseProps{
    courseName: string,
    onCourseDeleted: (courseName: string) => void;
}

const CourseCard: React.FC<CourseProps> = ({courseName, onCourseDeleted}) => {
  return (
    <div className='bg-[#2C3463] rounded-lg shadow-lg px-7 py-3 mr-5 flex transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]'>
        <div className='text-white text-center'>
        <Link href={`/collections?query=${courseName}`}>{courseName}</Link>
        </div>
        <div  onClick={() => onCourseDeleted(courseName)}><IoIosCloseCircleOutline className='ml-4 mt-0.5' color='white' size={20}/></div>
    </div>
  );
};

export default CourseCard;
