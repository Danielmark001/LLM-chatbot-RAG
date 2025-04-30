// components/Navigation.tsx

import React from 'react';
import Link from 'next/link';

/**
 * Navigation component for the application header
 */
const Navigation: React.FC = () => {
  return (
    <nav className='p-4 border-b-2 shadow-md fixed top-0 w-full z-10 bg-white'>
      <div className="container mx-auto flex items-center justify-between">
        <div>
          <Link href="/" className="text-[#2C3463] text-xl font-bold hover:text-[#3F50AD] transition-colors">
            Document Management System
          </Link>
        </div>
        <div className="flex gap-4">
          <Link 
            href="/"
            className="text-[#2C3463] hover:text-[#3F50AD] transition-colors"
          >
            Courses
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
