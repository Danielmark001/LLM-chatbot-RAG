"use client";

import React from 'react';
import { MdWavingHand } from "react-icons/md";
import { MdEmail } from "react-icons/md";
import { useRouter } from 'next/navigation';

const LoginContainer: React.FC = () => {
  const router = useRouter();
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className='w-1/4 rounded-lg shadow-lg p-5'>
        <h1 className="text-4xl font-semibold mb-2 text-[#3F50AD]">Login</h1>
        <h2 className="text-xl text-gray-500 mb-6">Admin Site</h2>
        <div className="flex items-center mb-4">
          <p className="text-lg text-gray-700 mr-2">Hi, Welcome</p>
          <MdWavingHand />
        </div>
        <div className="bg-[#3F50AD] text-white px-4 py-2 rounded w-full cursor-pointer" onClick={() => { router.push('/')}}>
            <div className='flex justify-center'>
            <MdEmail className='mr-2 size-6'/>
             Login with NTU Email
            </div>
        </div>
      </div>
    </div>
  );
};

export default LoginContainer;
