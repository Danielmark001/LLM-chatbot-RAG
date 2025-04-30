# Document Management System Admin Portal

This is the admin portal for the Document Management System, a platform for organizing and managing course materials with embedded search capabilities.

## Features

- Course management dashboard
- Document upload and management interface
- File preview functionality
- Responsive design with Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 16.x or higher
- npm or yarn
- Backend server running (see the flask-server directory)

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Create a `.env.local` file with your configuration:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Development Server

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

### Project Structure

- `app/page.tsx` - Main dashboard showing all courses
- `app/collections/page.tsx` - Document management interface for a specific course
- `app/components/` - Reusable UI components
- `app/login/` - Authentication interface

### Building for Production

```bash
npm run build
# or
yarn build
```

Then start the production server:

```bash
npm run start
# or
yarn start
```

## Technologies Used

- Next.js 13+ with App Router
- TypeScript
- Tailwind CSS
- React Icons
