import { useState } from 'react';
import { Oval } from 'react-loader-spinner';

interface DeletionPopupProps {
  onClose: () => void;
  onCourseDeleted: () => void;
  courseName: string
}

const DeletionPopup: React.FC<DeletionPopupProps> = ({ onClose, onCourseDeleted, courseName}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleDelete = () => {
    setIsLoading(true)
    fetch('http://localhost:5000/api/dropcollection', {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ collectionName: courseName }),
    })
    .then(response => {
      if (response.status === 201) {
        return  fetch('http://localhost:5000/api/dropIndex', {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ collectionName: courseName }),
        });
      } else if(!response.ok) {
        console.error('Failed to delete index');
      }
    })
    .catch(error => {
      console.error('Error deleting course:', error);
    })
    .finally(() => {
      setIsLoading(false);
      onClose();
      onCourseDeleted();
  });
}

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-gray-800 bg-opacity-50 z-50">
      <div className="flex flex-col bg-white p-8 rounded-lg shadow-md relative items-center">
        {isLoading && (
          <>
            <div className="flex justify-center mb-4">
              <Oval
                height={40}
                width={40}
                color="#2c4787"
                visible={true}
                ariaLabel='oval-loading'
                secondaryColor="#2c4787"
                strokeWidth={2}
                strokeWidthSecondary={2}
              />
            </div>
            <p className="text-[#1a2d58] text-center mb-4 font-semibold">Deleting</p>
          </>
        )}
        <p className='font-semibold text-lg'>Are you sure you want to delete the course?</p>
        <p className='text-[#ff3b3b] text-sm'>Warning: Deleting this folder will delete all the content inside</p>
        <div className='flex w-1/2'>
        <button
        onClick={handleDelete}
        className="bg-[#2C3463] text-white font-bold py-2 px-4 rounded mr-5 mt-5 w-2/5 transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
        >
        Yes
        </button>
        <button
        onClick={onClose}
        className="bg-[#2C3463] text-white font-bold py-2 px-4 rounded ml-5 mt-5 w-2/5 transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
        >
        No
        </button>


        </div>
      </div>
    </div>
  );
};

export default DeletionPopup;
