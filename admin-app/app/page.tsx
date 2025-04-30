"use client";

import { useEffect, useState } from "react";
import { MdDriveFolderUpload } from "react-icons/md";
import Popup from "./components/CollectionPopup";
import CourseCard from "./components/CourseCard";
import DeletionPopup from "./components/DeletionPopup";

/**
 * Main dashboard component displaying all available courses
 */
export default function Home(): JSX.Element {
  // State declarations
  const [collections, setCollections] = useState<string[]>([]);
  const [showPopup, setShowPopup] = useState<boolean>(false);
  const [showDeletionPopup, setShowDeletionPopup] = useState<boolean>(false);
  const [collectionCreated, setCollectionCreated] = useState<boolean>(true);
  const [collectionDeleted, setCollectionDeleted] = useState<boolean>(true);
  const [selectedCollection, setSelectedCollection] = useState<string>('');

  // Event handlers
  const handleCollectionCreated = () => {
    setCollectionCreated(!collectionCreated);
  };

  const handleCollectionDeleted = () => {
    setCollectionDeleted(!collectionDeleted);
  };

  const handleAddCourseClick = (): void => {
    setShowPopup(true);
  };

  const handleClosePopup = (): void => {
    setShowPopup(false);
  };

  const handleDeletePress = (courseName: string): void => {
    setSelectedCollection(courseName);
    setShowDeletionPopup(true);
  };

  const handleCloseDeletionPopup = (): void => {
    setShowDeletionPopup(false);
  };

  // Fetch collections on component mount or when collections are created/deleted
  useEffect(() => {
    fetchCollections();
  }, [collectionCreated, collectionDeleted]);

  /**
   * Fetch all available collections from the API
   */
  const fetchCollections = () => {
    fetch('http://localhost:5000/api/collections')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to fetch collections');
        }
        return response.json();
      })
      .then((collections: string[]) => {
        setCollections(collections);
      })
      .catch(error => {
        console.error('Error fetching collections:', error);
      });
  };

  return (
    <main className="flex h-screen flex-col p-24 bg-gray-100">
      {/* Header with title and add button */}
      <div className="flex flex-row justify-between">
        <div className="font-semibold relative w-10 text-xl">
          Courses
          <div className="absolute bottom-0 left-2 w-full h-1 bg-[#3F50AD]"></div>
        </div>
        <div>
          {collections.length > 0 && (
            <button 
              onClick={handleAddCourseClick} 
              className="bg-[#2C3463] text-white py-2 px-4 rounded-lg font-normal transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
            >
              Add New Course
            </button>
          )}
        </div>
      </div>

      {/* Display courses or empty state */}
      {collections.length > 0 ? (
        <div className="flex flex-wrap mt-5">
          {collections.map((collection: string, index: number) => (
            <CourseCard 
              key={index}
              courseName={collection} 
              onCourseDeleted={handleDeletePress}
            />
          ))}
        </div>
      ) : (
        <div className="flex h-screen flex-col items-center justify-center">
          <div className="flex flex-col bg-white w-4/5 border border-dotted border-[#3F50AD] p-4 mx-auto rounded-lg items-center">
            <MdDriveFolderUpload size={50} color="#2C3463" />
            <p className="font-semibold text-lg mt-2">Create a folder for the course</p>
            <button 
              onClick={handleAddCourseClick} 
              className="bg-[#2C3463] text-white py-2 px-4 rounded-lg font-normal mt-5 w-2/5 transition-transform duration-300 ease-in-out transform hover:scale-105 hover:bg-[#3C456C]"
            >
              Add New Course
            </button>
          </div>
        </div>
      )}
      
      {/* Modals/Popups */}
      {showPopup && (
        <Popup 
          onClose={handleClosePopup} 
          onCollectionCreated={handleCollectionCreated}
        />
      )}
      
      {showDeletionPopup && (
        <DeletionPopup 
          onClose={handleCloseDeletionPopup} 
          onCourseDeleted={handleCollectionDeleted} 
          courseName={selectedCollection}
        />
      )}
    </main>
  );
}
